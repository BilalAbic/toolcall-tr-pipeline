# 07 — Gerçek kaynak pilot kayıtları

Tarih: 2026-08-13

Bu belge yalnız revision, lisans, içerik hash'i, sayım ve immutable artifact
kimliklerini kaydeder. Ham kaynak satırı, credential veya provider isteği
içermez. Kaynak snapshotları `sources/`, türetilmiş JSONL ve pilot kanıtları
`artifacts/` altında yereldir ve kaynak denetimine dahil edilmez.

## Ortak sınırlar

- Her kaynak değişmeden okunmuş, pilot yayınından hemen önce yeniden hash'lenmiştir.
- Pilot zinciri: snapshot → strict ingest → canonicalize/karantina → exact audit.
- Pilot koşularında kayıt onarımı, insan adjudication, source-evidence promotion,
  S400 seçimi ve Gold/release yapılmamıştır. Pilotlardan sonra yalnız
  source-explicit, policy-covered, conflict-free/alias-dışı candidate cohort'a
  explicit provider egress'i yapılmıştır; bu koşular hiçbir pilot sonucunu,
  kararını veya üyeliğini değiştirmez.
- `blocked` sonucu bir hata gizlemez: aşağıdaki karantina veya conflict kanıtı
  çözülmeden sonraki kapı açılmaz.
- Bu kanıtlar 2026-08-13'te `review prepare` ile karar üretmeden sıralı yerel
  worklist'e dönüştürüldü; kaynak satırı, credential veya reviewer kararı
  worklist'e yazılmadı.

## Tarihsel iki-judge bounded automation koşusu

2026-08-13'te önceki iki-judge sürümündeki `automation run`, When2Call ve xLAM canonical artifactlarının
her birinde deterministic ilk 500 source-row penceresinden 50 source-explicit,
policy-covered, conflict-free/alias-dışı episode seçti. Bu pencere ve 600-leaf
üst sınırı candidate manifestine bağlandı; kaynak satırları değişmedi.

| Aşama | Sonuç |
|---|---|
| Candidate | 50 episode / 599 policy-izinli leaf; `autocand_7005053d…8a6ac` |
| Translation | 46 host-merged translation, 4 `needs_review`, 0 fallback route; `autobatch_736eb0cd…57f7d0` |
| Mini judge | 547 input; 538 result, 9 unavailable; `liveevalrun_6e8da414…05354` |
| Strong judge | 547 input; 513 result, 34 unavailable; `liveevalrun_a3f4a8a2…28b2` |
| Consensus | 456 two-judge `pass`, 91 `needs_review`; `autoconsreport_621881ee…a1a90` |
| HF review package | 12 strict `silver_candidate` JSONL satırı; `hfpackage_1af985da…7c9f7e`; `pending_human_approval` |

Paket yalnız yerel ignore edilmiş artifact kökündedir. `publish_allowed=false`
olduğu için Gold/release veya Hugging Face upload yapılmamıştır.

## Seçici-escalation canlı koşusu

2026-08-13’te aynı immutable canonical/audit girdileri üzerinde
`when2call-xlam-50-hierarchical-cost-20260813` ayrı bir output kökünde
çalıştırıldı. Bu koşu Flash → güvenli hatada Pro → tüm leaf’lerde mini judge →
mini non-pass ve mini-pass örneklemi için güçlü judge yolunu kullandı. Koşudaki
pass örneklemi o anki config ile `%10` idi; kodun sonraki varsayılanı maliyeti
azaltmak için `%2`dir. İki koşunun artifactleri birbirini değiştirmez.

| Aşama | Sonuç |
|---|---|
| Candidate | 50 episode / 524 policy-izinli leaf; `autocand_7005053d…8a6ac` |
| Translation | 45 host-merged translation, 5 `needs_review`, 0 Flash → Pro route |
| Mini judge | 524 input; 508 `pass`, 11 `fail`, 1 `needs_human_review`, 4 unavailable |
| Strong judge | 70 escalation: 54 mini-pass örneklemi + 16 mini non-pass; 54 `pass`, 3 `fail`, 4 `needs_human_review`, 9 unavailable |
| Hierarchical consensus | 508 accepted leaf, 16 `needs_review`; `autohconsreport_34ff5156…ffdc` |
| HF review package | 34 strict `silver_candidate` JSONL satırı; `hfpackage_7f135e7c…92887`; `pending_human_approval` |
| Token/maliyet | DeepSeek Flash `$0.032936`, GPT-5.4-mini `$0.354575`, GPT-5.4 `$0.196588`; toplam tahmini `$0.584098` |

Bu sonuçlar model quality proof’tur; `publish_allowed=false`, Gold/release veya
Hugging Face upload yapılmamıştır. Canlı inceleme, `reverse_input:/input_value`
ve `qrcode:/data` gibi doğal dil görebilen argument yollarının başka kayıtlarda
çok-türlü değer ya da identifier de taşıdığını ortaya koydu. Bu nedenle global
argument `copy_exact` politikası korunmuştur.

## Prompt v3 ve Flash 6-worker regresyonu

Bu regression’daki `translation-prompt-0.3.0`, daha kesin judge output kontratı,
varsayılan `%2` strong-pass örneklemi ve Flash için 6 worker ile aynı immutable
50-episode cohort üzerinde `when2call-xlam-50-prompt-v3-workers6-20260813`
ayrı output kökünde canlı doğrulandı. Kaynak, candidate manifesti ve önceki
artifactler değiştirilmedi.

| Aşama | Sonuç |
|---|---|
| Candidate | Aynı 50 episode / 599 policy-izinli leaf; `autocand_7005053d…8a6ac` |
| Translation | 46 host-merged translation, 4 `needs_review`; bir güvenli Flash → Pro fallback route (27 Pro leaf isteği) |
| Mini judge | 547 tamamlanmış structured verdict: 520 `pass`, 23 `fail`, 4 `needs_human_review` |
| Strong judge | 36 escalation: 27 mini non-pass + 9 `%2` pass örneklemi; 30 `pass`, 6 `fail` |
| Hierarchical consensus | 541 accepted leaf, 6 `needs_review`; `autohconsreport_c71c67d6…7631` |
| HF review package | 40 strict `silver_candidate` JSONL satırı, 10 episode dışarıda; `hfpackage_fe196677…7a50e`; `pending_human_approval` |
| Token/maliyet | Flash `$0.043040`, Pro `$0.007286`, GPT-5.4-mini `$0.409261`, GPT-5.4 `$0.096943`; toplam tahmini `$0.556529` |

Mini ve strong judge response-contract başarısı `547/547` ve `36/36` oldu;
önceki iki canlı koşuda görülen unavailable sonuç bu regression’da oluşmadı.
Bu bir kalite/maliyet kanıtıdır, otomatik Gold/release veya Hugging Face upload
yetkisi değildir.

Strong judge’ın altı doğrulanmış fail’i (gaz/benzin anlam daraltması, `null`
temsil ilişkisi, entity-address attachment, konum kapsamı ve iki akıcılık/terim
calque’u) `translation-prompt-0.4.0` için dar karşıt örneklere dönüştürüldü.
v0.4 sonraki ayrı immutable batch’te canlı doğrulanacaktır; bu mevcut kanıt
artifactini değiştirmez.

## NVIDIA When2Call

| Alan | Değer |
|---|---|
| Kaynak | `nvidia/When2Call`, tüm erişilebilir veri splitleri |
| Revision | `0582f7749df63a96fdc3070932e83e72396ace53` |
| Lisans | `CC-BY-4.0` |
| Toplam veri satırı | 27.952 |
| `test/llm_judge` | SHA-256 `sha256:0b710…3b25`; 300 satır → 296 canonical, 4 `TOOL_ARGUMENT_SCHEMA_INVALID`, conflict 0; `pilot_7fa0531624d313d378fe55815bb6827a2f594569a439d35b958796cb3f857ac6` |
| `test/mcq` | SHA-256 `sha256:8c369…b14c`; 3.652 satır → 3.541 canonical, 102 `TOOL_ARGUMENT_SCHEMA_INVALID`, 9 `SOURCE_ADAPTER_INVALID_FIELD`, 12 conflict; `pilot_cd080ac329066174de69909ffb73696a163047c3e6bf0ea978b4039319dbf265` |
| `train/pref` | SHA-256 `sha256:d9063…5bc4`; 9.000 satır → 8.787 canonical, 213 karantina (54 `TOOL_ARGUMENT_SCHEMA_INVALID`, 159 `SOURCE_ADAPTER_INVALID_FIELD`), 39 split-içi review candidate; `pilot_8efbdea27004207254412d7198d01a3cb7015f0ee1b18e9077ccaafa381749f6` |
| `train/sft` | SHA-256 `sha256:3eb20…631c`; 15.000 satır → 14.769 canonical, 231 `SOURCE_ADAPTER_INVALID_FIELD` karantina, 32 split-içi review candidate; `pilot_8e80681599c02200e4afbefd7904ae2c66d5ec513fe1d173f7f690b14d2cd80b` |
| Cross-split exact audit | 27.393 canonical survivor üzerinde 2.917 exact duplicate group, 136 review candidate; `audit_39ff84094f28f8e110f79d6ac1bbf3202a618ff5595f15507a7f288b20c311fb` |
| Sonuç | `blocked`: 559 karantina ve 136 conflict insan/kaynak kararı bekler |

`train/pref` seçilmiş assistant hedefini `chosen_response` alanında, `train/sft`
ise final assistant mesajında taşır. Tam `<TOOLCALL>…</TOOLCALL>` işareti
source-explicit tool çağrısıdır. Metin hedefleri yalnız açık ek-bilgi istemi veya
açık tool-unavailable ifadesi taşıdığında sırasıyla `clarification` veya
`tool_unavailable` olarak alınır. Belirsiz metin, bozuk APIGen tool şeması ve
geçersiz tool argümanı karantina kanıtı olarak kalır. İlk, tüm-training-karantina
pilotları korunmuş immutable geçmiş kanıttır; yukarıdaki v2 pilotları doğru
source target semantiğini temsil eder.

### Field-policy coverage

When2Call ve xLAM birlikte 85.111 canonical episode üzerinde ağsız policy
coverage koşusu yapıldı: 849.064 policy-izinli segment ve 0 unresolved policy
error. Tool/parameter açıklamaları çevrilebilir; argumentlar varsayılan olarak
`copy_exact` kalır ve provider'a gönderilmez.

## Salesforce xLAM Function Calling 60k

| Alan | Değer |
|---|---|
| Kaynak | `Salesforce/xlam-function-calling-60k`, dataset/train |
| Revision | `26d14ebfe18b1f7b524bd39b404b50af5dc97866` |
| Lisans | `CC-BY-4.0`; gated koşulları kullanıcı hesabında kabul edildi |
| Kaynak JSON SHA-256 | `sha256:4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6` |
| Kaynak JSON kayıt sayısı | 60.000 |
| JSONL türetme | `convert_af1ba3b5e9e95127be7830e2e261bf315fcfb2eff1193061187f741eab89dfc3` |
| Türetilmiş JSONL SHA-256 | `sha256:7e64730762a3c1dad8b7d385e99e2a670fb4ecc2b9edad14bbc37a638b8924d7` |
| Pilot | `pilot_c36575dddb9cb29db44b8a1abd8077771c777da421b2695d48ffc8d9911df61c` |
| Canonical | 57.718 |
| Canonical karantina | 2.282: 1.519 `TOOL_ARGUMENT_SCHEMA_INVALID`, 593 `SOURCE_ADAPTER_INVALID_FIELD`, 147 `TOOL_NAME_UNRESOLVED`, 23 `SCHEMA_CANONICAL_INVALID` |
| Exact conflict | 6 × unresolved `hard_conflict` / human review |
| Sonuç | `blocked`: karantina ve altı conflict çözülmeden selection/egress yok |

JSON dizi kaynak biçimi JSONL'e yalnız dışarıda, content-addressed ve
no-overwrite olarak türetildi. Dönüştürme kaynağın JSON hash'ini ve türetilen
JSONL hash'ini birleştirir; pilot snapshotı türetilmiş JSONL hash'iyle aynı
kimlik zincirine bağlanır.

## Yayımlanan insan review kuyruğu

`artifacts/review-queue/real-sources-20260813/` altında content-addressed
olarak yayımlanan worklist manifesti
`manifest_a52302ac6fac9d362db23109b2ee20bd762897ad653cf143d2c7b81ba9cc0c8d`'dir.
Beş güncel canonical-quarantine artifactı ile When2Call cross-split ve xLAM
exact audit'inden türetildi: 2.841 `remediate_canonical_quarantine` ve 142
`submit_conflict_adjudication` görevi, toplam 2.983 açık insan görevi vardır.
Bu, karar veya kaynak düzeltmesi değildir: conflict satırı yalnız haricî,
reviewer-authored `review submit-conflict` kararı hazırlanmasına yardım eder;
karantina satırı ise onaylı adapter/policy değişikliğinden sonra yeni immutable
pilot gerektirir.
