# SHAP & LIME ile Açıklanabilir Yapay Zeka Kredi Kartı Temerrüt Riski

Bir kredi kartı temerrüt tahmin modeli eğitip **aynı tahminleri** iki farklı açıklanabilirlik (XAI) yöntemiyle (**SHAP** ve **LIME**) açıklayan, çıktılarını karşılaştıran çalışma.


Blackbox bir modelin "neden bu kararı verdiğini" iki bağımsız yöntemle açıklamak ve yöntemlerin çıktılarını karşılaştırarak temel farklarını ortaya koymak.

## Veri Seti

**Taiwan Credit Card Default** (UCI / OpenML `data_id=42477`)

30.000 kredi kartı müşterisinin geçmiş ödeme davranışı, fatura tutarları ve demografik bilgileri.

| Hedef | Anlam |
|-------|-------|
| `0` | Ödendi: müşteri temerrüde düşmedi |
| `1` | Temerrüt: bir sonraki ay ödemesini yapmadı |

## Model

- **Algorithm:** Logistic Regression (`class_weight="balanced"`)
- **Precision:** `StandardScaler` ile standardizasyon
- Basit ve doğrusal açıklanabilirlik yöntemlerini net görmek için bilinçli tercih.

## Yöntemler

| **SHAP** (`LinearExplainer`) | Global + local katkılar | Özet grafik (summary) + tek müşteri waterfall |
| **LIME** (`LimeTabularExplainer`) | Tek tahminin yerel açıklaması | Tek müşteri HTML açıklaması |
| **Karşılaştırma** | Aynı müşteri için yan yana katkılar + yön uyumu | Karşılaştırma tablosu + diverging bar grafik |

## Notebook Akışı

1. **Veri & Model** veriyi yükle, ölçekle, Logistic Regression eğit
2. **SHAP** global özet + tek müşteri waterfall
3. **LIME** aynı müşterinin yerel açıklaması
4. **Karşılaştırma** SHAP ve LIME katkılarını yan yana tablo + grafik
5. **Fark Tablosu** iki yöntemin kavramsal karşılaştırması

## Temel Bulgular

- İki yöntem de en etkili değişken grubu olarak **ödeme gecikmelerini ve kredi limitini** öne çıkardı.
- Yüksek önemli değişkenlerde **yön uyumu** yüksekti; ayrışma düşük önemli değişkenlerde görüldü.
- Sebep: **SHAP deterministiktir** (aynı girdi -> aynı çıktı), **LIME ise rastgele örneklemeye dayalı bir yaklaşımdır** ve tekrarlarda hafif değişebilir.
- Sonuç çıkarım: Yüksek önemli değişkenlerde iki yöntem birbirini doğruluyorsa açıklamaya güvenilebilir.

## SHAP vs LIME 

| Boyut | SHAP | LIME |
|-------|------|------|
| Temel fikir | Oyun teorisi (Shapley) | Yerel doğrusal yaklaşım |
| Kapsam | Global + Local | Yalnızca Local |
| Teorik garanti | Var | Yok (yaklaşım) |
| Kararlılık | Deterministik | Değişebilir |
| Hız | Daha yavaş | Hızlı |
