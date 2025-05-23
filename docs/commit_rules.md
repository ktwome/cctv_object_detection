# 커밋 작성법

## 커밋의 구조
> [!QUOTE] 커밋의 구조
> 타입(스코프): 주제(제목) // Header(헤더)
> 본문 // Body(바디)
> 바닥글 // Footer

| 타입 이름    | 내용                               |
| -------- | -------------------------------- |
| feat     | 새로운 기능에 대한 커밋                    |
| fix      | 버그 수정에 대한 커밋                     |
| build    | 빌드 관련 파일 수정 / 모듈 설치 또는 삭제에 대한 커밋 |
| chore    | 그 외 자잘한 수정에 대한 커밋                |
| ci       | ci 관련 설정 수정에 대한 커밋               |
| docs     | 문서 수정에 대한 커밋                     |
| style    | 코드 스타일 혹은 포맷 등에 관한 커밋            |
| refactor | 코드 리팩토링에 대한 커밋                   |
| test     | 테스트 코드 수정에 대한 커밋                 |
| perf     | 성능 개선에 대한 커밋                     |

## 커밋하는 법

다음의 명령어를 이용하여 develope 브랜치에 커밋을 시작하세요.
```terminal
git checkout develope
git add .
git commit
```

이후 vim 편집기가 표시되는데, 사용법은 간단합니다.
a키를 눌러 수정 모드에 진입한 후, 윗줄에 '커밋 타입: 제목'을 입력하고,
세부 사항을 아래 줄부터 입력하시면 됩니다.
