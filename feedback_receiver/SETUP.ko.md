# PoreSAM 피드백 연결

피드백을 읽을 위치: [PoreSAM-feedback의 Issues](https://github.com/youngboong/PoreSAM-feedback/issues).
비공개 저장소는 이미 만들었습니다. 아래 설정은 소유자가 한 번만 하면 됩니다. 앱 사용자는 GitHub나 Google 로그인이 필요하지 않습니다.

## 1. GitHub 전용 토큰 만들기

[Fine-grained token 만들기](https://github.com/settings/personal-access-tokens/new)를 엽니다.

- Token name: `PoreSAM feedback`
- Resource owner: `youngboong`
- Repository access: **Only select repositories**, `PoreSAM-feedback`만 선택
- Repository permissions: **Issues — Read and write**, **Contents — Read and write**
- Expiration: 원하는 만료일. 만료되면 서버 설정에서 토큰을 교체해야 합니다.

토큰은 다음 단계의 **Script Properties**에만 넣습니다. 채팅이나 GitHub 파일에는 올리지 않습니다.

## 2. Google Apps Script에 서버 코드 넣기

[새 Apps Script 프로젝트](https://script.google.com/home/start)를 열고 제목을 `PoreSAM Feedback`으로 설정합니다.

1. 기본 `Code.gs`의 내용을 지우고 [서버 코드](Code.gs)를 전부 붙여넣고 저장합니다.
2. 왼쪽 **Project Settings**에서 **Script Properties → Add script property**를 엽니다.
3. 두 속성을 저장합니다.

| Property | Value |
| --- | --- |
| `GITHUB_REPOSITORY` | `youngboong/PoreSAM-feedback` |
| `GITHUB_TOKEN` | 1단계에서 만든 토큰 |

권한을 외부 HTTP 요청으로 제한하려면 **Show "appsscript.json" manifest file in editor**를 켜고, 편집기의 `appsscript.json` 내용을 [이 파일](appsscript.json)로 바꿉니다.

## 3. 배포하고 주소 보내기

오른쪽 위 **Deploy → New deployment**를 누릅니다.

- 유형: **Web app**
- Execute as: **Me**
- Who has access: **Anyone**

**Deploy**를 누르고 본인 계정에서 실행 권한을 승인합니다. 배포가 끝나면 **Web app URL**을 복사합니다. `/exec`로 끝나는 주소를 이 대화에 보내 주세요. 토큰은 보내지 않습니다.

그 주소를 앱에 연결한 뒤 실제 글·이미지 접수를 확인하고 설치본을 교체합니다. 이 단계가 끝나기 전에는 앱의 실제 전송이 연결되지 않습니다.

이후 피드백은 저장소의 **Issues**에서 읽고, 첨부 이미지 링크를 열어 확인합니다. 해결한 항목은 **Close issue**로 닫으면 됩니다.

[Google 배포 안내](https://developers.google.com/apps-script/guides/web), [GitHub 토큰 안내](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
