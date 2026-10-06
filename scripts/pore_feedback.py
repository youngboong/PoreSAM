"""Send only the message and images explicitly attached to Feedback."""
import base64
import binascii
from datetime import datetime, timezone
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid

from PIL import Image
from app_paths import ASSET_ROOT

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
IMAGE_FORMATS = {'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp', 'GIF': 'image/gif'}


def configuration():
    path = ASSET_ROOT / 'ui/feedback-config.json'
    values = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    endpoint = os.environ.get('PORESAM_FEEDBACK_ENDPOINT', values.get('endpoint', ''))
    if endpoint:
        url = urllib.parse.urlsplit(endpoint)
        if url.scheme != 'https' or not url.hostname or url.username or url.password or url.fragment:
            raise ValueError('Feedback requires a valid HTTPS receiver.')
    return dict(endpoint=endpoint, app_version=values.get('app_version', '2026.10.06'))


def submission(payload, app_version):
    message = payload.get('message', '')
    images = payload.get('images', [])
    if not isinstance(message, str) or len(message) > 5000:
        raise ValueError('Use up to 5,000 characters.')
    if not isinstance(images, list) or len(images) > 4:
        raise ValueError('Attach up to four images.')
    if not message.strip() and not images:
        raise ValueError('Write a message or attach an image.')
    attachments, total = [], 0
    for index, item in enumerate(images, 1):
        if not isinstance(item, dict) or not isinstance(item.get('data'), str):
            raise ValueError('Invalid image attachment.')
        if len(item['data']) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
            raise ValueError('Images must be under 5 MB each and 8 MB in total.')
        try:
            raw = base64.b64decode(item['data'], validate=True)
        except (ValueError, binascii.Error):
            raise ValueError('Invalid image attachment.') from None
        total += len(raw)
        if not raw or len(raw) > MAX_IMAGE_BYTES or total > MAX_TOTAL_BYTES:
            raise ValueError('Images must be under 5 MB each and 8 MB in total.')
        try:
            with Image.open(io.BytesIO(raw)) as image:
                mime = IMAGE_FORMATS.get(image.format)
                if not mime or image.width * image.height > 32_000_000:
                    raise ValueError('Use a PNG, JPG, WebP or GIF image under 32 megapixels.')
                image.verify()
        except (OSError, Image.DecompressionBombError):
            raise ValueError('Cannot read the attached image.') from None
        suffix = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif'}[mime]
        attachments.append(dict(name=f'image_{index}.{suffix}', mime=mime, data=item['data']))
    identifier = payload.get('id')
    try:
        identifier = str(uuid.UUID(identifier))
    except (ValueError, TypeError, AttributeError):
        raise ValueError('Invalid feedback identifier.') from None
    return dict(id=identifier, message=message.strip(), images=attachments,
                app_version=app_version, created_at=datetime.now(timezone.utc).isoformat())


def send_feedback(payload):
    config = configuration()
    record = submission(payload, config['app_version'])
    if not config['endpoint']:
        raise ValueError('The feedback inbox is not connected yet. Your message has not been sent.')
    request = urllib.request.Request(config['endpoint'], data=json.dumps(record).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.loads(response.read(16_000))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise ValueError('Could not send feedback. Check your connection and try again.') from None
    if not isinstance(result, dict) or result.get('accepted') is not True or result.get('id') != record['id']:
        raise ValueError('The inbox did not confirm receipt. Please try again.')
    return dict(sent=True, id=record['id'])
