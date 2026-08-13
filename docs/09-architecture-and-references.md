# 09 — Mimari değerlendirme ve kaynaklar

Tarih: 2026-08-13

Bu belge, projenin bütün yapısını inceler, kullanılan mimarinin tanınmış
standartlarla ilişkisini haritalar ve bu yapıya tam geçişin adımlarını verir.

## 1. Mimari değerlendirme

`toolcall-next` bir **provenance-first, tamper-evident, schema-driven,
fail-closed ML veri kümesi curation pipeline**'ıdır. Yani:

- **Provenance-first**: her çıktı, kaynağından bu yana geçtiği her dönüşümü
  hash'lenmiş kimlik zinciriyle taşır (`provenance.sources`, `snapshot_id` →
  `source_occurrence_id` → `episode_id` → `variant_id`, `parent_variant_id`).
- **Tamper-evident**: artifact'lar içerik-adreslidir (`publish_bytes_atomic`,
  hash ile adlandırılır), manifest'ler kendi gövdesinin SHA-256'sıdır ve
  `events.py` **append-only doğrulanmış hash zinciri** (blockchain-benzeri)
  tutar.
- **Schema-driven**: her JSONL satırı kendi `schema_version`'ını taşır; 101 adet
  Draft 2020-12 JSON Schema ile `StrictModel` (pydantic strict) üzerinden
  doğrulanır; bayt temsili RFC 8785 JCS ile kanoniktir.
- **Fail-closed**: belirsizlik asla sessizce düşürülmez; quarantine / review /
  `needs_review` ile insan kapısına bırakılır (`docs/02` A0–A13).

### Sonuç: bu yapı mantıklı mı?

**Evet.** Bu, yüksek bütünlüklü, denetlenebilir ve regülasyona duyarlı
(AB Yapay Zeka Yasası Madde 50 şeffaflık yükümlülüğü, NIST AI RMF) eğitim
verisi üretimi için bilinen ve savunulabilir bir paradigmanın doğrudan
uygulamasıdır. Kod zaten bu paradigmanın çekirdeğini (içerik-adresli depo +
doğrulanmış ledger + schema-first + insan-kapılı kalite) tutarlı biçimde
kurmuş. Eksik olan, bu iç yapının *dış standartlara göre adlandırılıp
dışa açılmasıdır* (bkz. §4).

## 2. Mekanizma → standart haritası

| İç mekanizma (`src/`, `docs/02`) | Karşılık geldiği standart | Durum |
|---|---|---|
| `artifacts.publish_bytes_atomic`, içerik-adresli dosya adları, `ContentManifest.contract_hashes` | **C2PA** claim/manifest + içerik bağlama (hash binding) | Uygulandı; imza yok |
| `events.EventLog` (previous_event_hash, sequence, event_hash) | **W3C PROV-DM** Activity/derivation zinciri + denetim günlüğü | Uygulandı |
| `CanonicalEpisode.provenance` (sources, pipeline, transformations) | **W3C PROV-DM** Entity/Activity/Agent; **RO-Crate** provenance | Uygulandı; PROV serileştirmesi yok |
| `docs/02` §4 ID-first zincir, `source_episode_fingerprint` | **W3C PROV-DM** derivations; **DVC** content-hash identity | Uygulandı |
| `canonical/`, `bronze/`, `quarantine/`, `manifests/`, disjoint output kökleri | **DVC** / **Pachyderm** immutable, reproducible data store | Uygulandı |
| `release_manifest`, `DATASET_CARD.md`, split/record-set tanımları | **MLCommons Croissant** (Dataset/Resource/Structure/Semantic katmanları) | Kısmi; Croissant JSON-LD export yok |
| `docs/`, `DATASET_CARD.md`, motivation/collection/limitations | **Datasheets for Datasets** / **Model Cards** | Kısmi |
| `field_policy.toml`, `configs/prompt_bundle.toml`, A0 contract freeze | **Croissant RAI**, **NIST AI RMF** governence | Uygulandı (içsel) |
| `StrictModel` + 101 Draft 2020-12 schema + RFC 8785 JCS | **JSON Schema 2020-12**, **RFC 8785** | Uygulandı |

## 3. Kaynaklar (kanonik referanslar)

### 3.1 İç tasarım referansları (kardeş depolar)

- [turkish-tool-calling-dataset](https://github.com/BilalAbic/turkish-tool-calling-dataset):
  aşamalı `S30 → S100 → S250 → S400` seçim, append-only ledger, teknik alanların
  deterministik korunması, çok kapılı kalite akışı.
- [magibu-toolcall](https://github.com/BilalAbic/magibu-toolcall): schema-first
  registry, canonical audit, provenance odaklı kayıtlar, insan kabulünün model
  kararından ayrı tutulması.

### 3.2 Dış standartlar

- **W3C PROV-DM** — Provenance Data Model (W3C Recommendation, 30 Apr 2013).
  <https://www.w3.org/TR/prov-dm/> · aile: PROV-O, PROV-N, PROV-CONSTRAINTS.
- **C2PA** — Coalition for Content Provenance and Authenticity, Technical
  Specification (v2.x). Tamper-evident, kriptografik olarak doğrulanabilir
  manifest'ler. <https://c2pa.org/specifications/> ·
  <https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html>
- **MLCommons Croissant** — ML-ready veri kümesi metadata formatı (schema.org
  üzerine JSON-LD; Dataset/Resource/Structure/Semantic katmanları + RAI
  uzantısı). Spesifikasyon: <https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html> ·
  repo: <https://github.com/mlcommons/croissant> · makale:
  <https://arxiv.org/abs/2403.19546> (NeurIPS 2024 D&B),
  DOI <https://doi.org/10.1145/3650203.3663326> · RAI:
  <https://mlcommons.org/croissant/RAI/1.0>.
- **RO-Crate** — Croissant ile aynı schema.org temelini paylaşan, genel amaçlı
  araştırma nesnesi paketleme/standardı (Croissant'a tamamlayıcı).
  <https://www.researchobject.org/ro-crate/> (spec 1.2:
  <https://www.researchobject.org/ro-crate/1.2/>).
- **Datasheets for Datasets** — Gebru et al., 2021.
  <https://arxiv.org/abs/1803.09010>.
- **Model Cards** — Mitchell et al., 2019. <https://arxiv.org/abs/1810.03993>.
- **DVC (Data Version Control)** — içerik-hash'li veri versiyonlama/hat.
  <https://dvc.org/doc>.
- **Hugging Face Datasets / Croissant desteği** — release export hedefi.
  <https://huggingface.co/docs/datasets> ·
  <https://huggingface.co/docs/datasets-server>.
- **RFC 8785** — JSON Canonicalization Scheme (JCS), bayt-kanonik temsil.
  <https://www.rfc-editor.org/info/rfc8785> (zaten kullanımda).
- **JSON Schema Draft 2020-12** — şema doğrulaması.
  <https://json-schema.org/draft/2020-12> (zaten kullanımda).
- **MQM (Multidimensional Quality Metrics)** — çeviri kalite çerçevesi.
  <https://www.jostrans.org/article/view/8074> (zaten kullanımda).
- **NIST AI Risk Management Framework** — governence / insan gözetimi.
  <https://www.nist.gov/itl/ai-risk-management-framework>.
- **EU AI Act, Article 50** — yapay zeka ile üretilmiş içeriğin şeffaflığı.
  <https://artificialintelligenceact.eu/article/50/>.

## 4. Tam geçiş yolu (gap analizi + yol haritası)

Mimari zaten "içsel" olarak bu standartları uyguluyor. Dışa açılma ve
tam uyum için beş faz:

### Faz A — Tamamlandı (içsel çekirdek)
İçerik-adresli depo, doğrulanmış hash-chain ledger, schema-first doğrulama,
fail-closed kalite kapıları, ID-first provenance zinciri.

### Faz B — Croissant export (önerilen sonraki adım)
`release_manifest` + `DATASET_CARD.md` + canonical JSONL'den bir
`croissant.json` (JSON-LD, schema.org `Dataset`) üret. Böylece Hugging Face /
Kaggle / OpenML araçları veri kümesini doğrudan "ML-ready" yükler.
- `RecordSet` = canonical episode satırları; `Field` = `conversation`,
  `tools`, `provenance`, `quality` vb.
- `distribution` = canonical JSONL + (varsa) Parquet.
- `split` = S30/S100/S250/S400 + train/val/test.
- RAI katmanı: lisans, insan-onay durumu, çeviri/gözden-geçirme kaynağı.

### Faz C — C2PA tarzı imzalama (önerilen, opsiyonel ama tavsiye)
Mevcut manifest'ler *tamper-evident* (hash zinciri) ama *signed* değil.
`ContentManifest` üzerine bir imza adımı (private-key, CAWG/C2PA sertifikası)
eklenirse kanıt non-repudiable (inkar edilemez) olur. Anahtar yönetimi ve
cihaz/CI imzalama politikası ayrıca tanımlanmalı.

### Faz D — PROV serileştirmesi (opsiyonel, birlikte çalışabilirlik)
`provenance` alanını PROV-JSON / PROV-O (RDF) biçimine dışa aktaran bir
yardımcı; böylece harici provenance araçları zinciri doğrudan tüketebilir.

### Faz E — Datasheets for Datasets tamamlama
`DATASET_CARD.md`'yi Gebru et al. şablonunun tüm bölümleriyle
(motivation, composition, collection, preprocessing, uses, distribution,
maintenance, ethical considerations) doldur ve Croissant RAI ile bağla.

## 5. Pratik sonraki adımlar (kod düzeyinde)

1. `scripts/export_croissant.py`: release manifest'ten `croissant.json` üretir
   (Faz B). `mlcroissant` ile doğrulanır.
2. `artifacts.py`: opsiyonel `sign_manifest(manifest, private_key)` yardımcısı
   (Faz C); imza manifest'e ayrı alan olarak eklenir, doğrulama
   `ContentManifest.validate_identity` içine girer.
3. `provenance.py` (yeni): `CanonicalEpisode.provenance` → PROV-JSON dönüşümü
   (Faz D).
4. `DATASET_CARD.md`: Datasheets şablonu bölümleri ile genişletilir (Faz E).

Hiçbiri mevcut fail-closed veya insan-kapılı davranışı değiştirmez; yalnız
iç yapıyı dış standartlara adlandırır ve dışa açar.
