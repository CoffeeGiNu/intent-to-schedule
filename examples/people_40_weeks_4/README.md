# people_40_weeks_4

Five synthetic teams of eight people over four weeks.

- `calendar.json`: input for `init --calendar`. People, working hours, and 810 existing appointments. The horizon runs from Monday 2026-10-12 to Saturday 2026-11-07 in `+09:00`, with 30-minute slots.
- `intents.en.txt`, `intents.jp.txt`: a scheduling survey and one reply per person, in English and Japanese. The replies are the requests to turn into commands. A few, such as a cap on new meetings per day, have no direct command.

Every team has the same eight roles. IDs are lowercase family names, such as `takahashi`.

| Role | Search | Recommendations | Checkout | Mobile | Platform |
|---|---|---|---|---|---|
| Team lead / project manager | Misaki Takahashi | Takuya Suzuki | Ayaka Shimizu | Hiroshi Nakajima | Naoko Kondo |
| Backend engineer (senior) | Kenta Ito | Yui Tanaka | Kenji Yamazaki | Mio Maeda | Shota Ishii |
| Frontend engineer | Sakura Yamamoto | Ren Watanabe | Nanami Mori | Kazuki Fujita | Miyu Sakamoto |
| Backend engineer (second year) | Sho Nakamura | Hina Saito | Sota Abe | Saki Ogawa | Yusuke Endo |
| Designer | Aoi Kobayashi | Mei Matsumoto | Yuna Ikeda | Tomoya Goto | Kana Aoki |
| QA | Daisuke Kato | Kaito Inoue | Riku Hashimoto | Erika Okada | Takumi Fujii |
| Data / machine learning | Yu Yoshida | Rina Kimura | Asuka Yamashita | Daiki Hasegawa | Rei Nishimura |
| SRE | Jun Sasaki | Yuto Hayashi | Haruki Ishikawa | Chihiro Murakami | Koji Fukuda |

Hours are 9:30 to 18:30 with lunch from 12:00 to 13:00. Designers work 10:00 to 16:00 and SREs 9:00 to 18:00. QA is away on 10/16, and data / machine learning on the afternoon of 10/21. Monday 10/12 and Tuesday 11/3 are holidays.

The search team's replies match `people_8_weeks_2` with deadlines one week later, plus a few meetings with other teams.
