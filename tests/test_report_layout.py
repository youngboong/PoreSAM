"""Report sizing and seamless image composition across export formats."""
import base64
import io
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_composer import selected,figure_scale,render_html,data_url,table_html
from report_hwpx import export_hwpx,NS


def asset(key,size,color,kind='image'):
    return dict(id=key,kind=kind,image=data_url(Image.new('RGB',size,color)))


def settings(*ids):
    return dict(title='SEM',conditions=[],method='Method',results='Results',extra_images=[],
                items=[dict(id='figure_test',selected=True,caption='Test',image_ids=list(ids),rows=[list(ids)],columns=1)])


class LayoutTest(unittest.TestCase):
    def test_different_aspects_touch_without_white_gutter(self):
        data=dict(assets=[asset('a',(100,300),'red'),asset('b',(400,100),'blue')])
        _,combined=selected(data,settings('a','b'))[0]
        image=Image.open(io.BytesIO(base64.b64decode(combined['image'].split(',')[1])))
        row=[image.getpixel((x,image.height//2)) for x in range(image.width)]
        first=next(i for i,pixel in enumerate(row) if pixel==(255,0,0))
        last=max(i for i,pixel in enumerate(row) if pixel==(0,0,255))
        self.assertNotIn((255,255,255),row[first:last+1])

    def test_histogram_and_mixed_figure_sizes(self):
        data=dict(assets=[asset('hist',(640,440),'red','histogram'),asset('image',(640,440),'blue')])
        histogram=selected(data,settings('hist'))[0][1]
        self.assertEqual(figure_scale(histogram),.77)
        self.assertIn('max-width:77%;max-height:500.5px',render_html(data,settings('hist')))
        mixed=selected(data,settings('hist','image'))[0][1]
        self.assertEqual(figure_scale(mixed),1.)
        image=Image.open(io.BytesIO(base64.b64decode(mixed['image'].split(',')[1])))
        row=[image.getpixel((x,image.height//2)) for x in range(image.width)]
        self.assertAlmostEqual(row.count((255,0,0))/row.count((0,0,255)),.77,places=2)
        self.assertEqual(selected(data,settings('image','hist'))[0][1]['kind'],mixed['kind'])

    def test_single_photo_scaling_and_table_rules(self):
        data=dict(assets=[asset('original',(640,440),'red'),asset('comparison',(1280,440),'blue'),
                         dict(id='stats',kind='table',headers=['Statistic','Width (um)'],rows=[['Mean','1.234'],['SD','0.123']])])
        self.assertEqual(figure_scale(selected(data,settings('original'))[0][1]),.75)
        self.assertEqual(figure_scale(selected(data,settings('comparison'))[0][1]),1.)
        self.assertIn('max-width:75%',render_html(data,settings('original')))
        self.assertIn('Width<br>(um)',table_html(data['assets'][2]['headers'],data['assets'][2]['rows']))
        options=settings('original');options['items'].append(dict(id='stats',selected=True,caption='Statistics'))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.hwpx';export_hwpx(path,data,options)
            with zipfile.ZipFile(path) as z:
                section=ET.fromstring(z.read('Contents/section0.xml'));head=ET.fromstring(z.read('Contents/header.xml'))
            borders={e.get('id'):e for e in head.findall('.//hh:borderFill',NS)}
            table=section.find('.//hp:tbl',NS);rows=table.findall('hp:tr',NS)
            for i,row in enumerate(rows):
                for cell in row.findall('hp:tc',NS):
                    border=borders[cell.get('borderFillIDRef')]
                    for side in ['left','right']:self.assertEqual(border.find('hh:'+side+'Border',NS).get('type'),'NONE')
                    self.assertEqual(border.find('hh:topBorder',NS).get('type'),'SOLID' if i==0 else 'NONE')
                    self.assertEqual(border.find('hh:bottomBorder',NS).get('type'),'SOLID' if i in [0,len(rows)-1] else 'NONE')

    def test_hwpx_histogram_size_and_spacing(self):
        data=dict(assets=[asset('hist',(640,440),'red','histogram'),asset('image',(640,440),'blue')])
        options=settings('hist');options['items']+=settings('image')['items'];options['items'][1]['id']='figure_other'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.hwpx';export_hwpx(path,data,options)
            with zipfile.ZipFile(path) as z:
                section=ET.fromstring(z.read('Contents/section0.xml'));header=ET.fromstring(z.read('Contents/header.xml'))
            pics=section.findall('.//hp:pic/hp:sz',NS)
            self.assertAlmostEqual(int(pics[0].get('width'))/int(pics[1].get('width')),.77/.75,places=4)
            self.assertAlmostEqual(int(pics[0].get('height'))/int(pics[1].get('height')),.77/.75,places=4)
            gap=next(p for p in header.findall('.//hh:paraPr',NS) if p.find('.//hh:lineSpacing',NS).get('type')=='FIXED')
            self.assertTrue(any(p.get('paraPrIDRef')==gap.get('id') for p in section.findall('hp:p',NS)))


if __name__=='__main__':unittest.main()
