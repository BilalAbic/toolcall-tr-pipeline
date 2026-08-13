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
  → GPT-5.4 leaf judge
  → both pass for every episode leaf
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
| Mini veya strong judge hata/fail/`needs_human_review` | Consensus `needs_review` | Dışarıda kalır |
| İki judge da her leaf için `pass` | `accepted_for_review_package` | Episode HF JSONL’ye girer |

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
  --episodes 1000 --max-segments 10000 --workers 4 --live
```

`--source-row-cap` yalnız düşük maliyetli canary’dir; manifestte açıkça
kaydedilir. Tam üretimde verilmez. `--workers` 1–16 arası bağımsız leaf/episode
çağrısını paralelleştirir; artifact kimliklerini veya sırayı değiştirmez.

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

## 50 episode canlı kanıtı

`when2call-xlam-50-20260813` koşusu, her canonical inputun ilk 500 source
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
Bu kanıt Gold, release veya HF publish değildir.
