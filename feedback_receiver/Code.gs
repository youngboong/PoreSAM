// Deploy as a Google Apps Script web app: execute as owner, access Anyone.
// Keep GITHUB_TOKEN in Script Properties, never in the desktop application.
function doGet() {
  return jsonResponse_({service: 'PoreSAM feedback'});
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  if (!lock.tryLock(10000)) return jsonResponse_({accepted: false});
  try {
    var p = PropertiesService.getScriptProperties();
    var repo = p.getProperty('GITHUB_REPOSITORY');
    var token = p.getProperty('GITHUB_TOKEN');
    if (!repo || !/^[\w.-]+\/[\w.-]+$/.test(repo) || !token) throw new Error('Receiver not configured');
    var payload = JSON.parse(e.postData.contents);
    if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(payload.id)) throw new Error('Invalid identifier');
    if (typeof payload.message !== 'string' || payload.message.length > 5000 || !Array.isArray(payload.images) || payload.images.length > 4) throw new Error('Invalid feedback');
    if (!payload.message.trim() && !payload.images.length) throw new Error('Empty feedback');
    var cached = CacheService.getScriptCache();
    if (cached.get(payload.id)) return jsonResponse_({accepted: true, id: payload.id});
    var marker = '<!-- poresam-feedback:' + payload.id + ' -->';
    // Check a durable idempotency marker before writing images or creating an issue.
    var existing = github_(repo, token, 'issues?state=all&per_page=100', 'get');
    if (existing.some(function(issue) {return (issue.body || '').indexOf(marker) >= 0;})) {
      cached.put(payload.id, 'accepted', 21600);
      return jsonResponse_({accepted: true, id: payload.id});
    }
    var date = Utilities.formatDate(new Date(), 'UTC', 'yyyy-MM-dd');
    var dayKey = 'daily_' + date;
    var used = Number(p.getProperty(dayKey) || '0');
    if (used >= Number(p.getProperty('DAILY_LIMIT') || '50')) throw new Error('Daily limit reached');
    var total = 0;
    var links = payload.images.map(function(image, index) {
      var extensions = {'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/gif': 'gif'};
      var extension = extensions[image.mime];
      if (!extension || typeof image.data !== 'string' || image.data.length > 6990508) throw new Error('Invalid image');
      var bytes = Utilities.base64Decode(image.data);
      total += bytes.length;
      if (!bytes.length || bytes.length > 5242880 || total > 8388608) throw new Error('Image limit exceeded');
      var path = 'attachments/' + date + '/' + payload.id + '/image_' + (index + 1) + '.' + extension;
      var file = github_(repo, token, 'contents/' + path, 'get', null, true);
      if (!file) {
        file = github_(repo, token, 'contents/' + path, 'put', {message: 'Feedback image ' + payload.id, content: image.data}).content;
      }
      return '[Image ' + (index + 1) + '](' + file.html_url + ')\n\n![Image ' + (index + 1) + '](' + file.html_url + '?raw=true)';
    });
    var text = payload.message.trim();
    var title = text ? text.split(/\r?\n/)[0].slice(0, 100) : 'Image feedback';
    var body = marker + '\n' + text;
    if (links.length) body += '\n\n### Images\n\n' + links.join('\n\n');
    body += '\n\n---\nPoreSAM ' + String(payload.app_version || '').slice(0, 40) + ' · ' + date;
    github_(repo, token, 'issues', 'post', {title: title, body: body});
    p.setProperty(dayKey, String(used + 1));
    cached.put(payload.id, 'accepted', 21600);
    return jsonResponse_({accepted: true, id: payload.id});
  } catch (error) {
    // Never return the token, GitHub response body, or account details to the app.
    return jsonResponse_({accepted: false});
  } finally {
    lock.releaseLock();
  }
}

function github_(repo, token, path, method, data, missingAllowed) {
  var options = {method: method, muteHttpExceptions: true,
    headers: {'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}};
  if (data) {options.contentType = 'application/json'; options.payload = JSON.stringify(data);}
  var response = UrlFetchApp.fetch('https://api.github.com/repos/' + repo + '/' + path, options);
  var code = response.getResponseCode();
  if (missingAllowed && code === 404) return null;
  if (code < 200 || code >= 300) throw new Error('GitHub request failed');
  return JSON.parse(response.getContentText());
}

function jsonResponse_(value) {
  return ContentService.createTextOutput(JSON.stringify(value)).setMimeType(ContentService.MimeType.JSON);
}
