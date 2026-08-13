# 08 — Otomatik adaydan Hugging Face review paketine

Tarih: 2026-08-13

Bu akış, tek tek insan kararı beklemeden çalışır. İnsan yalnız en sonda,
hashlenmiş HF review paketini inceleyip yayın kararı verir. Hiçbir otomatik
çıktı Gold veya publish yetkisi taşımaz.

## Akış

```text
immutable canonical + exact audits
  → deterministic source-explicit/conflict-free candidate cohort
  → DeepSeek Flash translation
  → safe terminal failure: one DeepSeek Pro fallback
  → host merge + source rehash
  → GPT-5.4-mini leaf judge
  → mini non-pass + deterministic mini-pass sample: GPT-5.4 strong judge
  → unsampled mini pass OR strong final pass for every episode leaf
  → HF JSONL review package
  → explicit human approval
  → separate Gold/release/HF publish decision
```

Her aşama disjoint output köküne content-addressed JSON/JSONL artifact yazar.
Kaynak, canonical kayıt ve önceki artifactler yerinde değiştirilmez.

## Otomatik karar matrisi

| Olay | Otomatik davranış | HF review paketine etkisi |
|---|---|---|
| Flash geçerli çeviri | Host-side merge, checkpoint | Judge aşamasına gider |
| Flash `provider_response_invalid`, `malformed_response`, `response_too_large` veya HTTP transient | Bir kez Pro fallback | Pro geçerse judge aşamasına gider |
| Ağ teslimatı belirsiz | Yeniden gönderme yok; `needs_review` | Dışarıda kalır |
| Preflight block | Transport yok; `needs_review` | Dışarıda kalır |
| `research_needed` | Bir kez Pro fallback | Hâlâ çözülmezse dışarıda kalır |
| Mini judge `pass`, örneklem dışı | Mini kararı finaldir | Leaf kabul edilir |
| Mini judge `pass`, deterministik örneklem içinde | Güçlü judge’a escalation | Güçlü `pass` ise kabul edilir; diğer sonuçta dışarıda kalır |
| Mini judge hata/fail/`needs_human_review` | Güçlü judge’a escalation; yeniden çeviri yok | Güçlü `pass` ise kabul edilir; diğer sonuçta dışarıda kalır |
| Güçlü judge hata/fail/`needs_human_review` | Yeniden gönderme yok; consensus `needs_review` | Dışarıda kalır |

Fallback yalnız yeni model route’u olarak yazılır; ham prompt, model yanıtı veya
credential artifact’e girmez. Aynı immutable route yeniden başlatılırsa mevcut
receipt/checkpoint kullanılır.

## Çalıştırma

Tam üretim için 1.000 episode ve açık leaf bütçesi kullanılır:

```powershell
uv run tcdata automation run <canonical-jsonl>... `
  --conflict-audit <audit.json>... `
  --output <disjoint-output> `
  --config configs/provider-smoke.toml `
  --episodes 1000 --max-segments 10000 --strong-pass-sample-percent 2 --live
```

`--source-row-cap` yalnız düşük maliyetli canary’dir; manifestte açıkça
kaydedilir. Tam üretimde verilmez. Provider config, Flash için 6, mini judge
için 4, güçlü judge için 2 eşzamanlı worker tanımlar; `--workers` ile 1–16
arasında açık override verilebilir. Artifact kimlikleri ve sıra değişmez.
`--strong-pass-sample-percent` varsayılan `2`’dir. Bu yüzde yalnız mini
`pass` kayıtlarının deterministik ikinci-değerlendirici örneklemidir; mini
non-pass kayıtlar her zaman güçlü judge’a gider. Güçlü judge hiçbir zaman
çeviri fallback’i tetiklemez.

2026-08-13 operator dashboard limitleri `gpt-5.4-mini` için 200k TPM / 500
RPM / 2M TPD, `gpt-5.4` için 500k TPM / 500 RPM / 900k TPD'dir. Proje bunların
altında başlangıç ayarı kullanır: mini 4 worker ve 1.8M günlük uyarı eşiği;
güçlü judge 2 worker ve ücretsiz paylaşım kotasına göre 225k eşik. OpenAI
hesap/proje bazlı gerçek rate-limit başlıklarını döndürdüğü için sabit kamuya
açık RPM varsayımı yapılmaz. DeepSeek'in resmi sınırları Flash için 2.500,
Pro için 500 eşzamanlı istektir; 6 worker bu sınırların çok altındadır.

## Canlı durum ve maliyet

CLI seçim, çeviri, mini judge, seçici strong escalation ve final consensus
aşamalarını görünür biçimde basar. Her provider yanıtındaki token
sayaçlarından şu tablo oluşur: provider, model, dış istek sayısı,
input/cache/output tokenı, fiyatlanmış yanıt sayısı ve tahmini USD tutarı.
Ham payload, yanıt, header veya credential kaydedilmez. Fiyat tahmini fatura
değil, çalıştırma anındaki sabit fiyat kartıdır; provider güncel faturası
otoritedir.

## HF review paketi

Her paket şu dosyaları üretir:

```text
hf-review-package/<package-id>/
  data/train.jsonl
  dataset_info.json
  README.md
  manifest.json
```

`data/train.jsonl` her satırda strict `id`, OpenAI/TRL-uyumlu `messages`,
`tools`, kaynak namespace/snapshot kimliği, `silver_candidate` tier’i ve
`human_approval_required=true` taşır. `manifest.json` input ve çıktı
SHA-256’lerini bağlar; durum her zaman `pending_human_approval`,
`publish_allowed=false` olarak başlar.

## Tarihsel iki-judge 50 episode canlı kanıtı

`when2call-xlam-50-20260813` önceki iki-judge koşusu, her canonical inputun ilk 500 source
satırından 50 deterministic aday ve 599 leaf seçti. Sonuçlar:

- 46 episode host-side merge ile çevrildi; 4 episode preflight sonrası
  `needs_review` olarak korundu.
- 547 tam-leaf judge inputu üretildi.
- Mini judge: 538 sonuç / 9 sonucu olmayan receipt.
- Strong judge: 513 sonuç / 34 sonucu olmayan receipt.
- 456 leaf iki judge tarafından `pass` oldu; 91 leaf `needs_review` kaldı.
- Bütün leafleri uzlaşan 12 episode, 12 satırlık upload-ready JSONL review
  paketine girdi.

Paket ID:
`hfpackage_1af985da8eb55f39112c4123ab67e427ba37d2998ccf496ffab2ad7d7b7c9f7e`.
Bu tarihsel kanıt Gold, release veya HF publish değildir; yeni seçici-escalation
stratejisi kendi immutable koşusunda ayrı kanıt üretir.

## Seçici-escalation 50 episode canlı kanıtı

`when2call-xlam-50-hierarchical-cost-20260813`, aynı canonical/audit inputları
ve ayrı output köküyle 50 deterministic aday seçti. Bu koşuda o an `%10` strong
pass örneklemi kullanıldı; güncel varsayılan `%2`dir. Sonuçlar:

- 45 episode host-side merge ile çevrildi; 5 episode `needs_review` olarak
  korundu; Flash → Pro fallback route oluşmadı.
- 524 full-leaf mini judge inputu üretildi: 508 `pass`, 11 `fail`, 1
  `needs_human_review`, 4 unavailable.
- 70 leaf strong judge’a çıktı: 54 mini-pass örneklemi ve 16 mini non-pass.
  Strong 54 `pass`, 3 `fail`, 4 `needs_human_review`, 9 unavailable verdi.
- Hierarchical consensus 508 leaf’i kabul etti, 16 leaf’i `needs_review` tuttu;
  tüm leafleri kabul edilen 34 episode strict HF review paketine girdi.
- Sağlayıcı sayaçlarından tahmini maliyet: Flash `$0.032936`, mini `$0.354575`,
  strong `$0.196588`; toplam `$0.584098`.

Paket ID:
`hfpackage_7f135e7c5e6a7f33c2a4280732e456ff8fbb2248bb7c77db4ebfe30511092887`.
Bu kanıt da Gold, release veya HF publish değildir: paket
`pending_human_approval` ve `publish_allowed=false` durumundadır.

## Prompt v3 ve Flash 6-worker regresyonu

`when2call-xlam-50-prompt-v3-workers6-20260813`, aynı 50 episode cohort'u
ayrı immutable output kökünde güncel prompt, Flash 6 worker ve `%2` strong-pass
örneklemiyle tekrar çalıştırdı. 46 episode çevrildi, 4 episode `needs_review`
olarak korundu ve 1 güvenli Flash → Pro fallback route kullanıldı.

- Mini judge 547/547 geçerli structured verdict üretti: 520 `pass`, 23 `fail`,
  4 `needs_human_review`.
- Strong judge 36 leaf’e çıktı: 27 mini non-pass ve 9 pass-örneklemi; 30
  `pass`, 6 `fail` verdi. Hiç provider-response unavailable oluşmadı.
- Hierarchical consensus 541 leaf’i kabul etti, 6 leaf’i `needs_review` tuttu;
  tüm leafleri kabul edilen 40 episode strict HF review paketine girdi.
- Tahmini maliyet Flash `$0.043040`, Pro `$0.007286`, mini `$0.409261`, strong
  `$0.096943`; toplam `$0.556529`.

Paket ID:
`hfpackage_fe1966778e00f1c0a16fd0b955ed369f302f2c90c671bddfeb2226038637a50e`.
Paket `pending_human_approval` ve `publish_allowed=false` durumundadır; insan
onayı olmadan Gold/release/Hugging Face publish yapılamaz.

Güçlü judge’ın altı gerçek fail bulgusu `translation-prompt-0.4.0` adayına dar
karşıt örnekler olarak eklendi. Aynı 50-episode cohort’taki ayrı immutable v0.4
regression’u 535 accepted leaf ve bir `provider_response_invalid` unavailable
güçlü-judge sonucu verdi; doğrulanmış v0.3’ün 541 accepted leaf / 0 unavailable
sonucunu geçemediği için promotion reddedildi. Tarihsel artifact saklanır fakat
aktif prompt yeniden v0.3’tür.

Operational translation, prompt bundle’daki `promotion_status` alanını transport
öncesinde denetler: yalnız `validated` canlı egress başlatabilir; `candidate` ve
`retired` durumları fail-closed reddedilir. Böylece geliştirme adayları yeni bir
immutable regression ve açık promotion kararı olmadan ana üretime giremez.
