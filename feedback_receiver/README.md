# GitHub feedback inbox

The desktop Feedback window submits text and pasted or selected images to this receiver. The receiver creates GitHub Issues and stores attachments in a separate feedback repository. Users do not need a GitHub account. The desktop never contains a GitHub access token.

## Connect once

For this installation, `youngboong/PoreSAM-feedback` has already been created as a private repository. Follow the [Korean setup steps](SETUP.ko.md) and return only the deployed `/exec` URL. Never share the token in chat.

1. Create a private repository for feedback, such as `PoreSAM-feedback`. Initialize it with a README and enable Issues.
2. Create a fine-grained GitHub token restricted to that repository, with **Issues: Read and write** and **Contents: Read and write**.
3. Create a Google Apps Script project and paste [Code.gs](Code.gs).
   You can use [appsscript.json](appsscript.json) to limit OAuth access to external HTTP requests.
4. Under **Project Settings → Script Properties**, set `GITHUB_REPOSITORY` to `owner/PoreSAM-feedback` and `GITHUB_TOKEN` to the token. Paste the token there, not into the desktop config or Git.
5. Choose **Deploy → New deployment → Web app**. Execute as yourself and set access to **Anyone**. Authorize the script and copy its `/exec` URL.
6. Set `endpoint` in [feedback-config.json](../ui/feedback-config.json) to that URL and build the app.

Read submissions under **Issues** in the feedback repository. Image links open the attached files; private repository access remains limited to its members. Close an issue when it is resolved.

The receiver accepts four images, up to 5 MB each and 8 MB total, and messages up to 5,000 characters. `DAILY_LIMIT` is an optional script property and defaults to 50 submissions per day. Only explicit attachments and the app version are sent. Analysis projects and logs are not attached automatically.

See the official [Apps Script deployment guide](https://developers.google.com/apps-script/guides/web) and [GitHub token guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
