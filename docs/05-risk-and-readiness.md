# 05 — Riskler ve üretime hazırlık

Bu belge, planın en kritik arıza biçimlerini görünür tutar ve bir sonraki faza geçmeden önce kontrol edilecek kısa listeleri tanımlar. Ayrıntılı süreçleri tekrar etmez; `02-data-pipeline.md` ve `04-implementation-plan.md` kapılarına bağlanır.

Güncel karar: **Faz 1–5 temel, sınırlı canlı provider yüzeyi, iki gerçek kaynağın modelsiz pilotu, karar üretmeyen insan-review worklist'i ve en çok 30 episode'luk pre-review canary yüzeyi hazırdır; bu insan review kararı veya release'e geçiş onayı değildir.** Pre-review canary, teknik/prompt/provider testini insan kabulundan önce yapabilir; insan review S400/Gold/release için son karar kapısıdır. Varsayılan config çevrimdışıdır; canlı komutlar explicit `--live`, non-default config, preflight ve disjoint output gerektirir. Aşağıdaki listelerde test/model kanıtı, modelsiz gerçek kaynak kanıtı ve henüz açık üretim koşulları ayrılır.

## 1. Öncelikli risk kaydı

| Risk | Erken sinyal | Zorunlu önlem |
|---|---|---|
| Canonical schema fazla gevşek veya fazla katı | çok sayıda beklenmeyen pass/quarantine | strict fixture, property-based test ve sürümlü migration |
| Kaynak yanlış tool'u çağırıyor | query ile tool açıklaması uyuşmuyor | çeviri öncesi source semantic gate + insan adjudication |
| Argument kaynakta temelsiz | provenance `unknown` veya evidence pointer boş | otomatik valid'i engelle; `SOURCE_ARG_NOT_GROUNDED` |
| Geçerli alternatif davranış conflict sanılıyor | aynı context için iki makul karar | yalnız source/human yetkili `acceptable_behaviors` |
| Çeviri teknik alanı değiştiriyor | preservation hash farkı | leaf-segment üretimi + programatik merge + %100 invariant |
| Translation memory yanlış bağlamda reuse yapıyor | aynı metin, farklı tool/policy | exact tam-segment + context/policy fingerprint; fuzzy reuse yok |
| Prompt değiştiği hâlde eski gate sonucu kullanılıyor | config hash farklı, status aynı | attempt ve gate'i sıfırla; immutable config ID |
| Judge deterministik kontrolün yerini alıyor | schema/invariant hatası modelce pass | deterministik kapı önce ve bağlayıcıdır |
| Loss mask yanlış tokenları eğitiyor | role/target sınırı fixture'dan sapıyor | model-family render + token/loss-mask golden test |
| Hedef veya tool schema sessiz truncate oluyor | render token sayısı limitte | fail-closed length gate; silent truncation yok |
| JSONL rerun duplicate veya overwrite üretiyor | aynı kimlik iki kez yayınlanıyor | anti-join, append-only event ve atomic content-addressed publish |
| Lisans/terms snapshotı güncel değil | revision veya license hash değişmiş | source manifest pinleme ve release öncesi yeniden doğrulama |
| Research egress güvensiz | private IP, redirect veya aşırı içerik | pre-egress scan, allowlist, MIME/size/time/query budget |
| İnsan review tutarsız | annotator anlaşmazlığı yükseliyor | kör review, rubrik sürümü, calibration ve adjudication |

## 2. Faz 1–5 hazırlık listesi

Fixture/test ile doğrulanan temel:

- [x] Canonical, bronze, source snapshot, diagnostic, diagnostic catalog, run event ve content manifest şemaları `schemas/0.1.0/` altında Draft 2020-12 olarak üretilebilir.
- [x] Physical occurrence, episode, tool ve variant ID'leri deterministiktir.
- [x] Fixture ingest'te her geçerli ve hatalı fiziksel satır valid/quarantine olarak muhasebeleştirilir.
- [x] Tool normalizer'ın sıra ve keyword semantiği property-based testlidir.
- [x] Bilinmeyen diagnostic kodu ve desteklenmeyen schema keyword'ü fail-closed reddedilir.
- [x] Rerun farklı içeriğin mevcut artifact'ı overwrite etmesine izin vermez; aynı input aynı manifesti üretir.
- [x] Varsayılan configte providerlar ve network egress kapalıdır; canlı translation/judge yalnız explicit kapılar, allow-list transport, preflight ve hash-only provenance ile çalışır.
- [x] Source evidence Pass 1 yalnız explicit JSON Pointer kanıtını kabul eder; `unknown` ve `must_not_infer` otomatik kabulü engeller.
- [x] Exact/near duplicate ve conflict yüzeyi otomatik drop yapmaz; karar zinciri insan yetkilidir.
- [x] Split guard, deterministic reserve queue ve S30/S100/S250 strict S400 prefixleri sentetik adaylarla test edilmiştir.
- [x] Field policy yalnız açıkça izinli doğal dil leaf'lerini segmentleştirir; teknik alanlar ve policy'siz argumentler fail-closed kalır.
- [x] Provider-shaped segment yanıtı sentinel sırası, NFC ve tam coverage için yerelde doğrulanır; canlı DeepSeek/OpenAI adapterları yalnız sabit sentetik smoke ile kanıtlanmıştır.
- [x] Pre-egress scan secret/PII/local path/private endpoint bulgularını güvenli kayda indirger ve offline configte temiz payload'ı dahi bloklar.
- [x] MQM atomic finding/segment kanıtı ile Wilson %95 coverage raporu deterministiktir; model triage sonucu human review olmadan gold yapamaz.
- [x] Prompt katmanları immutable/content-addressed'dir; recorded-response fake provider, bozuk veya contract dışı yanıtı ağ açmadan reddeder.
- [x] Segment memory yalnız exact source/context/policy/tool/action anahtarıyla human-promoted entry döndürür; iki aktif hedef automatic seçim yerine conflict üretir.
- [x] Terminology risk routing ve research metadata public-HTTPS/bütçe sınırını doğrular fakat fetch yapmaz.

Gerçek kaynak üzerinde tamamlanan kanıt:

- [x] NVIDIA When2Call tüm erişilebilir splitler revision-pinned pilot: 27.952 kaynak-valid satır, 27.393 canonical, 559 gerekçeli karantina; dört splitte 2.917 exact duplicate group ve 136 human-review conflict adayı. Eğitim split'lerindeki seçilmiş no-tool hedefleri dahil edilir; belirsiz hedefler fail-closed karantinadadır.
- [x] Salesforce xLAM 60k train revision-pinned pilot: 60.000 kaynak-valid satır, 57.718 canonical, 2.282 gerekçeli karantina, 6 insan-review hard conflict adayı.
- [x] Her iki pilot kaynak değişiminde fail-closed rehash, immutable artifact publish ve kaynak-provider egress'i olmadan çalıştı.
- [x] Karar üretmeyen `review prepare` worklist'i, güncel beş canonical-quarantine artifactı ve iki exact audit'ten 2.841 karantina + 142 conflict görevi yayımladı; otomatik drop veya kabul yok.
- [x] Pre-review canary yalnız source-explicit, policy-covered, conflict-free ve exact-alias dışı kayıtları en çok 30 episode ile sınırlar. When2Call'da `translation-prompt-0.2.0` ile 3 episode / 20 leaf çevrildi; GPT-5.4-mini triage 20/20 pass, 0 finding verdi. Çıktılar Gold-ineligible kalır.

Gerçek kaynak üzerinde açık kapı:

- [ ] Hazır 2.983 görevlik worklist'teki When2Call 559 / xLAM 2.282 karantina ve When2Call 136 / xLAM 6 unresolved conflict, bütün pre-review canary kanıtları tamamlandıktan sonra yetkili insan tarafından kaynak kurallarına göre son kabulda adjudicate edilmeli; hiçbir kayıt otomatik onarılmamalı.
- [ ] Her canonical araç argümanı için gerçek source-evidence Pass 1 ve insan `source_valid` kararları üretilmeli.
- [ ] Faz 4–7 S400 membership, gerçek renderer/loss-mask ve insan review kapıları tamamlanmalı.

## 3. S400 seçim dondurma listesi

- Tüm kaynaklar A7 Pass 1'den geçmiş.
- Exact duplicate, conflict, near-duplicate ve split-leakage raporları üretilmiş.
- Reserve queue deterministik; seçim sırası ve gerekçesi manifestte.
- S400'deki her episode insan tarafından `source_valid` kabul edilmiş.
- Her expected argument provenance'a sahip; unresolved `unknown` veya `must_not_infer` yok.
- Alternatif davranışların yetkisi `source_explicit` veya `human_adjudicated`.
- Unresolved conflict ve quarantine membership içinde yok.
- S30, S100 ve S250, dondurulmuş S400'ün strict prefix kümeleri.

## 4. Çeviri config promotion listesi

- System prompt, modeller, generation params, schema, kod ve policy tek immutable config ID altında pinli.
- Dev/canary/frozen eval koşularında translation memory kapalı.
- Memory yalnız frozen sonuçtan sonra insan kararıyla promote ediliyor.
- Secret/PII/local-path kontrolü her provider ve research egress'inden önce çalışıyor.
- Unresolved/conflicting research sonucu accepted olamıyor.
- Structural preservation ve source/target segment coverage %100.
- Mini ve güçlü judge metrikleri insan gold setinde ayrı raporlanmış.
- Token ve maliyet bütçesi çağrıdan önce kontrol ediliyor.

## 5. S30/S100/S250/S400 kapı listesi

- İşlenen episode ID kümesi expected set ile birebir; missing ve extra sıfır.
- Model-generated ve memory-reused sonuçlar ayrı sayılıyor.
- Token, maliyet, retry, repair, research ve insan zamanı raporlanıyor.
- Hata matrisi source/action/tool/domain/uzunluk/provenance dilimlerini kapsıyor.
- Config ID değişmişse eski gate sonucu yeniden kullanılmıyor.
- İlk 400 için mini judge, güçlü judge ve insan kabulü tam kapsamlı.

## 6. Release hazırlık listesi

- Yalnız insan kabul edilmiş satırlar gold membership'e giriyor.
- Hedef model ailelerinde chat-template render ve loss-mask testi geçiyor.
- Hedef mesaj veya gerekli tool schema truncate edilmiyor.
- Canonical JSONL schema, full-row JCS hash, ordered dataset hash ve row count doğrulanıyor.
- Parquet istenirse JSONL↔Parquet full-row deep equality geçiyor.
- Release manifesti source, schema, prompt, model, code, lockfile ve review sürümlerini pinliyor.
- PII/secret/local-path ve lisans/provenance kontrolleri geçiyor.
- Dataset Card, xLAM tool-call-only satırlarının `awaiting_tool` olduğunu ve gold/silver ayrımını açıkça anlatıyor.
- `1.0.0` yalnız S400 gold, migration provası ve bilinen uyumsuzluklar kapandıktan sonra veriliyor.
