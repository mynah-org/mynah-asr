# ARCHIVE_INDEX (DRAFT, not uploaded) -- gabrione/mynah-asr-canary-it-experiments

Draft of 2026-10-27 from the read-only inventory (`.work/hf-archive-inventory-2026-10-27.md`). Proposed tiers are
NOT applied: every row still exists on the Hub. When pruning is approved, update `retention` and `on Hub`, then
upload this file as `ARCHIVE_INDEX.md` at the repo root.

Billed storage at draft time: 72.08 GB (`used_storage`); history squashed last on 2026-10-20 (6829b2c079).

Retention classes: KEEP-FULL (weights + record), KEEP-EVIDENCE (record kept; weights only if needed),
DELETE-CANDIDATE (tier 1 = safe, tier 2a / 2b = user judgement). Parent `stock` = parakeet_realtime_eou_120m-v1
(sha256 prefix 6603a22a, not on this Hub).

## Checkpoints

| experiment | status | checkpoint sha256 | canonical path | size GB | retention | proposed | parent | children | successor | cited by |
|---|---|---|---|---|---|---|---|---|---|---|
| FR-H2 plain parent | IMPORTANT (release parent) | `d2062aca28d8db0abff823fe4be8051665bdb4c7d3d15cc8ff929f32cf8bfda6` | `runs/ada-fr-h2-g1e-w4-23762/final.nemo` | 0.460 | KEEP-FULL | keep | stock | ada-fr-s2v4-h2 | - | EVO,P:S5-FR-prereg,RF22 |
| RC-FR v1.1 (FINAL-FR-v2 23.00) | RELEASE | `3130534d577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c` | `runs/ada-fr-s2v4-h2/final.nemo` | 0.460 | KEEP-FULL | keep | ada-fr-h2-g1e-w4-23762 | - | - | EVO,RF22 |
| AR-A2-R seed 0 | IMPORTANT (best AR model; primary band NULL) | `11b8790e6db5a3cd18e2e22a1bc4f4d0efaa960ddf55465d0751edd3f86ee290` | `runs/dgx-ar-a2-r-s12670/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | P:AR-D2-prereg,P:AR-SEED1-prereg,R24 |
| ES-D2 seed 1 | CURRENT (FINAL-ES candidate; best DEV 14.81) | `5e89541aa38c55fb9f326137460ef09f29ac983f24f7caeecc420a819f4b61d4` | `runs/dgx-cfs-d2-s1-es/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |
| ES-H3 seed 1 | CURRENT (FINAL-ES candidate) | `5b24cfe80aa46f322da552f6c097ad484ca96cadc0765ff62098209127a227f7` | `runs/dgx-cfs-h3-s1-es/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |
| ES-S5 seed 0 | CURRENT (preferred compute recipe; M1/M2 model) | `583d6954a285d85194029e7fd93c9957b92b9a8b9058ca81b47e59b7a74683c2` | `runs/dgx-cfs-s5-es/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | ES-H3 / ES-D2 | ESQplan,P:ES-S2Q-prereg,P:SUB15-ESD2-prereg,P:SUB15-M1M2-prereg,R24,R26,S14,SUB15plan |
| ES-S5 seed 1 | CURRENT (preferred compute recipe; M1/M2 model) | `df7ae15e4193321449defd149695b20bfe474601b727a82f20f84500535d8919` | `runs/dgx-cfs-s5-s1-es/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | ES-H3 / ES-D2 | ESQplan,P:SUB15-M1M2-prereg,R26,S14,SUB15plan |
| ES-D2 seed 0, 246 h | CURRENT (FINAL-ES candidate; best DEV) | `fb886e09eb7a34182b95457c07f0b33486658c885eff807419e2ee8d946a7218` | `runs/dgx-es-d2-23762/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | P:ES-S2Q-prereg,P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |
| ES-H3 seed 0 (SUB15) | CURRENT (FINAL-ES candidate; Stage-2 parent) | `1c28256c929d6b5fc5685ba854488ad69ea801b61b95d7ebb1414720c18ea900` | `runs/dgx-es-h3-23762/final.nemo` | 0.460 | KEEP-FULL | keep | stock | dgx-es-s2v4-h3 | - | P:ES-S2Q-prereg,P:FINAL-ES-prereg,P:SUB15-ESD2-prereg,R26 |
| ES Stage-2 v4 on H3 (ES-S2Q) | CURRENT (MARGINAL; open preservation-line RCA) | `beeef283a31a5d68b4d7b44c068b88b5f7e4f8022ef925a91fe3da392619da77` | `runs/dgx-es-s2v4-h3/final.nemo` | 0.460 | KEEP-FULL | keep | dgx-es-h3-23762 | - | - | P:ES-S2Q-prereg,R26 |
| ES-1 (172b1539) | IMPORTANT (factory reference) | `172b1539f5927f7f1b08cebb3f5c02a3011f89ed368a4cb5b5525dca58444638` | `runs/dgx-es1-w4-12670/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | ES-S5 / ES-H3 / ES-D2 | ESQplan,EVO,P:S5-prereg,P:SUB15-M1M2-prereg,R24,R26,RF22,S14,SUB15plan |
| S5-FR arm B (same-GPU G1E baseline) | IMPORTANT (factory compute evidence) | `ef840fd4e074731e6bf9e0824e6e74e9ab0b66c96a8be78d49ba70f9f5426743` | `runs/dgx-s5fr-b-g1e-11881/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | P:S5-FR-prereg,R25 |
| S5-FR arm S5 | IMPORTANT (factory compute evidence, STRONG) | `51db5578541b8e2f681dec5399ee27d5b29e2238501d16d55e5a0da01f21e467` | `runs/dgx-s5fr-s5-g1e-11881/final.nemo` | 0.460 | KEEP-FULL | keep | stock | - | - | EVO,P:S5-FR-prereg,R25 |
| FR-G1E plain parent | IMPORTANT (release parent) | `1271deff6a833805b5dc9fb021fba43c058782e686842a49718f4e362b030973` | `runs/jp-fr-g1e-w4-11881/final.nemo` | 0.460 | KEEP-FULL | keep | stock | jp-fr-s2v4-g1e, l40s-fr-s2v2-ps | FR-H2 | AUT,EVO,P:S5-FR-prereg,RF22,RIF |
| RC-FR v1 = FR-S2 (FR release, FINAL-FR 27.76) | RELEASE | `7814141d515ca347004a862435fc5029d553d885252c8ac5e5b23ce793d48105` | `runs/jp-fr-s2v4-g1e/final.nemo` | 0.460 | KEEP-FULL | keep | jp-fr-g1e-w4-11881 | - | RC-FR v1.1 candidate | AUT,EVO,RF22,RIF |
| IT-C250-36k plain parent | IMPORTANT (release parent) | `7e919281193735210f46a37985263e858b31a2d92621ac4378efd5cfb243a798` | `runs/plain-it-c250-36k/final.nemo` | 0.460 | KEEP-FULL | keep | stock | sg-it-s2v4-c250-36k, sg-it-s2v5b-c250-36k | - | EVO,RF22,RIF,RM12,RM15 |
| IT-E3 plain (whole encoder 2e-4) | IMPORTANT (parent of former production EOU-v3) | `a74468820a14b0744c7df379bbaea0264c66f6ef34b2e212809db992f9a1a134` | `runs/plain-it-e2-enc2e-4-s4000/final.nemo` | 0.460 | KEEP-FULL | keep | stock | plain-it-e3r1-lr1e4, plain-it-s2-e3, plain-it-s2-e3-djlr, plain-it-s2-e3-pre | IT-E3-8k / C-line | BN,RM12 |
| EOU-v3 (production until RC-IT) | IMPORTANT (former production anchor, RIF anchor 3) | `57d04dc4244322a57b4260c04354133dc80862c7f8c8652df0ebf4c9d27c9407` | `runs/plain-it-s2-e3-djlr/final.nemo` | 0.460 | KEEP-FULL | keep | plain-it-e2-enc2e-4-s4000 | - | RC-IT v4 | BN,RIF,RM15 |
| RC-IT v4 (IT release, FINAL-IT 20.49) | RELEASE | `0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff` | `runs/sg-it-s2v4-c250-36k/final.nemo` | 0.460 | KEEP-FULL | keep | plain-it-c250-36k | - | - | EVO,RF22,RIF,RM15 |
| AR-D2-R seed 0 | HARM (cv_decon +2.10) | `069c1ee05b1918bb028296e2c25c1501c59f250d0716bc6645c1da380ca8a281` | `runs/dgx-ar-d2-r-s12670/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | AR-A2-R | P:AR-D2-prereg,P:AR-SEED1-prereg,R24 |
| AR-D2-R seed 1 | HARM REPLICATED | `c53209a177ca22346c84158d491aabbb85272e4b7a41973553c1920dfbcef1c3` | `runs/dgx-ar-d2-r-s12670-seed1/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | AR-A2-R | P:AR-SEED1-prereg |
| IT DGX lab R0 control | NULL (lab control) | `a5f5311003a6f914ee8a38c6bfffd3197aac8eed2cfc048ff65ec0ef11b2f8bc` | `runs/dgx-r0-ctl-1k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | - | RM15 |
| IT DGX lab R2 CTC 0.1 | NULL | `ec2dfc0541036171c42d80fab27f4062d7a72ce27b885686d7bdd7b7cd4d78ce` | `runs/dgx-r2-ctc01-1k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | - | RM15 |
| FR-G1 | HARM (+1.75) | `de457d0b6501e7f4bf2332cf91cf3fc0683891e482a18d2885b1af3d23a71982` | `runs/jp-fr-g1-w4-8k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | FR-G1E | RIF,RM15 |
| FR-v2 stage 2 (pause stretch) | FAILED | `0817c2f1da6a4aec7260aab23024a77ccb2cb5ece3670afe70dbc7835c793085` | `runs/l40s-fr-s2v2-ps/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | jp-fr-g1e-w4-11881 | - | - | AUT |
| IT tournament S0 (cosine 2k control) | SUPERSEDED (screen) | `696c3ac5c7463a9c6459ac060677f216b88b37e2a905e5f258c8ddbde681eb37` | `runs/l40s-s0-cos2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT tournament S1 (WSD 2k) | SUPERSEDED (screen) | `415cd21d00662f69994d68be83aac004013e6a8c6fa67f83e0509d89ede34e54` | `runs/l40s-s1-wsd2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT tournament S2 (enc x2, 2k) | SUPERSEDED (screen) | `668c959085ad453729656c77af9c4d3b6273f18f6e718c3510082292b3b7de60` | `runs/l40s-s2-enc2x2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT tournament S3 (2k) | SUPERSEDED (screen) | `e6c7569fd55bc68cb4562966f53fbe50f071715d8ff88291d7c6b5045b4dc2ae` | `runs/l40s-s3-wsd-enc2x2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | IT-S3-4k | - |
| IT tournament S4 (enc 8e-4, 2k) | SUPERSEDED (screen) | `b92b5c80e0dc0f285edc0addead9f455a9e522aa4ad4f270e94449ae78f46006` | `runs/l40s-s4-wsd-enc8e4-2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT tournament S5 (encoder 8e-4, 4k) | NULL (flat) | `d14ce0120d1d3d12d2382dc761c3121731e30d9f737a538cb52a1f80cec4fca2` | `runs/l40s-s5-wsd-enc8e4-4k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT tournament S6 (enc 1.6e-3 boundary probe) | FAILED (overshoot probe) | `8dd43942eac7647051b0a6a82253e513422a72fe726ae37479e5bd20e0029069` | `runs/l40s-s6-wsd-enc16e4-2k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | W4 | - |
| IT-E2 LR sweep (whole encoder 1e-4) | SUPERSEDED (sweep arm) | `f5ab05fbbcd2852fbf57a764b9231495acfe634c3fe67e543e2fd527341caf00` | `runs/plain-it-e2-full1e4/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | IT-E3 | - |
| IT Italian SPE on B2 | NULL/HARM (confounded negative) | `cd4ea4767b191ba034a41792fe9422013ac10ceb88d01768930e8a121cea3dea` | `runs/plain-it-tok-b2/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | stock | - | - | - |
| IT stage-2 v5-B | FAILED (pre-registered latency FAIL) | `ab84cd3be6a0abf32f31bbb50c7675fe623cdd6fcd4587383e5c04fdf276e38e` | `runs/sg-it-s2v5b-c250-36k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2a | plain-it-c250-36k | - | RC-IT v4 | RIF |
| Canary-180M IT B4 40 h (lightweight track 2026-10-09) | SUPERSEDED (other model family, closed track) | `3196d670634911ebe5c1f71235d9c80eb63591a7419e6e9e77ec1a2e5e744b85` | `runs/B4-40h-e7/final.nemo` | 0.741 | KEEP-EVIDENCE | tier 2b | canary-180m-flash | - | - | LWR |
| IT-W4-C250-16k | SUPERSEDED (PARTIAL vs 36k); was the frozen scorer of the closed D-selection axis | `b13e4e7e8ad0e44accf8eff6dfb41e5afa9a5fd491b246e27047356a3f7b2073` | `runs/ada-it-w4-c250-16k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250-36k | RM15 |
| AR-A1-R | SUPERSEDED (GO-FULL, replaced by A2) | `b981e33c69d94c95c08fb9150f9f1b023ff50610f79ff48f6d305c6a186d8c33` | `runs/dgx-ar-a1-r-s4000/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | AR-A2-R | EVO,P:AR-A2-prereg,P:AR-D2-prereg,R24 |
| AR-A2-R seed 1 | NULL (seed replicate) | `0024c4ed9169f238642bbc3ff00bf4511321037896a324e41bdd1641d7002232` | `runs/dgx-ar-a2-r-s12670-seed1/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | AR-A2-R s0 | P:AR-SEED1-prereg |
| ES-S2 seed 0 (bucketing) | SUPERSEDED (STRONG, replaced by S5) | `3f50100b6ab23e17ccdc9d28d770e8b0a31aec6e8cd870bfb59bebe4929de97c` | `runs/dgx-cfs-s2-es/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | ES-S5 | S14 |
| ES-S2 seed 1 | SUPERSEDED (STRONG, replaced by S5) | `5e51e0fdcf22caee62abbc6e83f131216dc710997f97f928c71e9d601f684d80` | `runs/dgx-cfs-s2-s1-es/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | ES-S5 | P:S5-prereg,S14 |
| FR-E1 | SUPERSEDED (CLEAR, G1E chosen) | `3a2ebe311e9293cff4dca5df3a8debb55728323919d9eec3ec95ce3a1bec3858` | `runs/jp-fr-e1-w4-11881/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | FR-G1E | RIF |
| FR-P1 | SUPERSEDED (LEVER -38, then E1/G1E) | `1ec3fec4e1784e668638d0526fdd51141ff4678ddc2abd56a5ed6dcd1513880c` | `runs/jp-fr-p1-w4-8k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | FR-E1 / FR-G1E | RIF,RM15 |
| IT-W4-C250-8k | SUPERSEDED (screen) | `7635d9cfc9bee3543e25bcce35dcf71592e2f3b4c8af47108cd1d8c8b4233ec1` | `runs/l40s-it-w4-c250-8k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | W4-C250-16k | RIF,RM15 |
| IT-S3-4k (origin of W4) | SUPERSEDED (milestone) | `5c10f0982768d2246af4b9260d856c11f06ebb68e94b74dd9e3ba6933f45c6a2` | `runs/l40s-s3-wsd-enc2x-4k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | factory W4 | - |
| FR-0 old p40 | SUPERSEDED (harness-fr 65.85) | `e23b11e2459bc042b228bf0222fb7b2b9c73ec8b8db71ab067afa802a591a6de` | `runs/plain-fr-p40/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | plain-fr-s2-eou | FR-P1 | FT,RIF,RM15 |
| FR-0 stage 2 (s2-eou) | SUPERSEDED | `5266c90f6302922f9ec04ed454f93ad9405ecfd7bab4e4bd3ef8797ac67ad054` | `runs/plain-fr-s2-eou/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | plain-fr-p40 | - | RC-FR v1 | FT,RIF,RM15 |
| IT-B2 plain | SUPERSEDED | `995270c888d91e49d043af1a50201eeea1728477f496099ebdf96cd25ba0229d` | `runs/plain-it-b2-gradual/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | plain-it-e1a-decjoint, plain-it-e1b-all1e4, plain-it-s2-b2, plain-it-s2-b2-full, plain-vpchk-b2-tail | IT-E3 | - |
| IT-C250 14k | SUPERSEDED (by C250-36k) | `28cbfc954acfc44db03495b2ef1a72395b08f410d9be055423d83db085c9873f` | `runs/plain-it-c250/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250-36k | EVO,RIF,RM12,RM15 |
| IT-C250-20k | SUPERSEDED (horizon ladder) | `031cfdaf11df9ba50534e45e4f05eae55935ae9d65b06490ac022d4a0fb0d2a4` | `runs/plain-it-c250-20k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250-36k | RM12 |
| IT-C250-28k | SUPERSEDED (horizon ladder) | `9129842ff85f513e340fad770c364ae7d7d7a7a9b93029cb2067038cdc9eed73` | `runs/plain-it-c250-28k/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250-36k | RM12 |
| IT-C250-oldrep | SUPERSEDED (exposure test, closed) | `b17362c12753473f0b4d9a12daea69f0f1a560521e37ae773ac20dcda480253d` | `runs/plain-it-c250-oldrep/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250-36k | EVO,RM12 |
| IT-E3-8k | SUPERSEDED (horizon lever) | `0090f15bf8d0af49bc330bc1b577d64b15f8cecce5fdd29f9c82d2fe37d875f7` | `runs/plain-it-e3-s8000/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C-add25 / C250 | RM12 |
| IT-E3 R1 re-warm | SUPERSEDED (best plain on 10-12) | `3e6009551ab9715d76814acf056b5bfce17f47036bd7a715e4deafaad7336ba0` | `runs/plain-it-e3r1-lr1e4/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | plain-it-e2-enc2e-4-s4000 | - | C250 line | RM12 |
| IT-C-add25 | SUPERSEDED (by C250) | `7ee0b8f2101264d03e203e445e981827ca2f5876141646f6db60ebb1477fe643` | `runs/plain-it-p1-add25/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | stock | - | C250 | EVO,RM12 |
| EOU-IT v2 (stage 2 on B2) | SUPERSEDED (by EOU-v3) | `63e5305aa1272fe9fc6cbdd99a7137796f0a329aca3a65ed1c415bddd39dfeb0` | `runs/plain-it-s2-b2/final.nemo` | 0.460 | KEEP-EVIDENCE | tier 2b | plain-it-b2-gradual | - | EOU-v3 | MD |

Note: parent / children names come from `metrics.json` `args.init` (box paths `plain-<tag>`); the Hub dir of a parent
is `runs/<box>-<tag>` (e.g. `plain-it-c250-36k` -> `runs/plain-it-c250-36k`, `plain-fr-g1e-w4-11881` ->
`runs/jp-fr-g1e-w4-11881`, `plain-es-h3-23762` -> `runs/dgx-es-h3-23762`).

## Other weight files

| artifact | sha256 | canonical path | size GB | retention | note |
|---|---|---|---|---|---|
| RC-IT v4 Mynah pack | `2aa4783d566880a67e8694cd39ba4eeb84b8dc724ea047f2f82d5c78da3df79f` | `rc/it-v4/mynah-pack/model.safetensors` | 0.459 | KEEP-FULL | byte-identical copy in release/mynah-eou-it-120m/mynah/ |
| RC-FR v1 Mynah pack | `9e0e6b50a20f3e4f655c5916092bf523ce5730317da89fdae7b3f5bba73dbfa8` | `rc/fr-s2v4/model.safetensors` | 0.459 | KEEP-FULL | byte-identical copy in release/mynah-eou-fr-120m/mynah/ |
| RC-FR v1.1 Mynah pack | `28e3f0f54a9d1c2a48278b08a5dea7d7b4a63abb03b42d6f808c0bce8b8daf05` | `release/mynah-eou-fr-120m-v1.1/mynah/model.safetensors` | 0.459 | KEEP-FULL | export of 3130534d |
| ES-1 Mynah pack | `dd54fb207d927745c163873f4f771bbe10b4ea974511ff7656ccb7543836e5bf` | `runs/dgx-es1-w4-12670/mynah-pack/model.safetensors` | 0.459 | KEEP-FULL | export of 172b1539 |
| IT-E3 trainer state | `9c41098daecd5f7cbd37c534689e2a1e2235c45c14ad11ff090ecfe52f9713a3` | `runs/plain-it-e2-enc2e-4-s4000/last.ckpt` | 1.379 | DELETE-CANDIDATE tier 1 | optimizer state of a CLOSED run |
| RC-IT v4 release copy | `0a4d1b103297e169af74b6673ff97864747bb2c54b70a7ec91253d9583c003ff` | `release/mynah-eou-it-120m/model.nemo` | 0.460 | KEEP-FULL | = runs/sg-it-s2v4-c250-36k/final.nemo |
| RC-FR v1 release copy | `7814141d515ca347004a862435fc5029d553d885252c8ac5e5b23ce793d48105` | `release/mynah-eou-fr-120m/model.nemo` | 0.460 | KEEP-FULL | = runs/jp-fr-s2v4-g1e/final.nemo |
| RC-FR v1.1 release copy | `3130534d577e340df129fea9074a63a070aaa05292bcf3aadd96b4aa39fd969c` | `release/mynah-eou-fr-120m-v1.1/model.nemo` | 0.460 | KEEP-FULL | = runs/ada-fr-s2v4-h2/final.nemo |

## Data

| artifact | canonical path | size GB | retention | dependencies |
|---|---|---|---|---|
| it-c250 | `data/snapshots/it-c250` | 23.880 | KEEP-FULL | training data of C250-36k -> RC-IT v4 (restore: data_snapshot.py restore --name it-c250) |
| fr-g1e | `data/snapshots/fr-g1e` | 14.799 | KEEP-FULL | training data of FR-G1E / FR-H2 -> RC-FR v1 / v1.1 (restore --name fr-g1e) |
| final-fr | `data/final-fr` | 0.831 | KEEP-FULL | FINAL-FR sealed bank (spent, RC-FR v1) |
| final-it | `data/final-it` | 0.762 | KEEP-FULL | FINAL-IT sealed bank (spent, one shot of RC-IT v4) |
| final-it-wip | `data/final-it-wip` | 0.699 | DELETE-CANDIDATE tier 1 | superseded by data/final-it (strict subset) |
| final-fr-v2 | `data/final-fr-v2` | 0.567 | KEEP-FULL | FINAL-FR-v2 sealed bank (spent, RC-FR v1.1) |
| ood-fr1 | `data/ood-fr1` | 0.182 | KEEP-FULL | FR-OOD1 sealed bank (E1 vs G1E decision) |
| eou-fr-v2-dev | `data/eou-fr-v2-dev` | 0.056 | KEEP-FULL | DEV-EOU-FR-v2 |
| eou-fr-v2-final | `data/eou-fr-v2-final` | 0.051 | KEEP-FULL | FINAL-EOU-FR-v2 (sealed) |
| it | `data/it` | 0.041 | KEEP-FULL | IT teacher pools |
| a2-fr | `data/a2-fr` | 0.034 | KEEP-FULL | A2-fr manifests |
| smoke-pool | `data/snapshots/smoke-pool` | 0.020 | KEEP-FULL | - |

## Records (always KEEP-FULL)

`recipe/` (the exact scripts), `tokenizers/`, `logs/`, `results/` (dated result dirs, survival tarballs, local-only
git bundles), every `runs/*/metrics.json`, prereg copies, logs, evals, hypotheses and gate outputs, including the
weightless and smoke run dirs.
