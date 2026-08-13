# Prompt dizini

Bu dizin Faz 5 için ayrılmıştır. Etkin ve content-addressed prompt katmanları `configs/prompt_bundle.toml` içindedir; ayrı bir prompt metin dosyası şu anda yoktur. DeepSeek çeviri ve OpenAI judge adapterları bu bundle'ı yalnız explicit canlı kapılar arkasında kullanır.

Field policy, segment extraction/merge, sentinel koruması, pre-egress taraması, strict output schema, fake-provider/recorded-response testleri ve sürüm/hash sözleşmesi uygulanmıştır. Buna rağmen gerçek kaynak içeriği, insan-review ve selection kapıları tamamlanmadan provider'a gönderilmez; API anahtarı prompt dosyasına yazılmaz.

Gelecekte her prompt immutable sürümlü bir alt dizinde tutulmalı; exact bytes, output schema ve bağlı policy/config hash'leri her attempt ile kaydedilmelidir. Secret değerleri promptlara veya bu dizine yazılmaz.
