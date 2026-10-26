import os
import time
import urllib.request
import json

papers = [
    {
        "id": "2211.15979",
        "filename": "01_AirFormer_Predicting_Nationwide_Air_Quality_Transformers_AAAI23.pdf",
        "title": "AirFormer: Predicting Nationwide Air Quality in China with Transformers",
        "authors": "Yuxuan Liang, Yutong Xia, Songyu Ke, Yiwei Wang, Qingsong Wen, Junbo Zhang, Yu Zheng, Roger Zimmermann",
        "year": 2023,
        "category": "Air Quality Transformers & Spatiotemporal Modeling"
    },
    {
        "id": "2502.17919",
        "filename": "02_AirCast_Air_Pollution_Forecasting_Multivariable_Alignment.pdf",
        "title": "AirCast: Improving Air Pollution Forecasting Through Multi-Variable Data Alignment and fMAE Loss",
        "authors": "Vishal Nedungadi, Muhammad Akhtar Munir, Salman H. Khan, et al.",
        "year": 2025,
        "category": "Rare Event / Heavy-Tailed Loss & ViT for Air Quality"
    },
    {
        "id": "2606.07648",
        "filename": "03_AQIFormer_Transformer_Multiview_CrossCity_Air_Quality.pdf",
        "title": "AQIFormer: A Transformer-Based Multi-View Architecture for Cross-City Air Quality Classification",
        "authors": "Tanmay Sarkar, et al.",
        "year": 2026,
        "category": "Cross-City Transfer & Transformer for AQI"
    },
    {
        "id": "2002.12898",
        "filename": "04_PM25_GNN_Domain_Knowledge_Enhanced_GNN.pdf",
        "title": "PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting",
        "authors": "Shuo Wang, Yanbo Jiang, et al.",
        "year": 2020,
        "category": "Physics/Domain-Knowledge Spatio-Temporal Graph Neural Networks"
    },
    {
        "id": "2301.13629",
        "filename": "05_DiffSTG_Probabilistic_Spatiotemporal_Graph_Diffusion.pdf",
        "title": "DiffSTG: Probabilistic Spatio-Temporal Graph Forecasting with Denoising Diffusion Models",
        "authors": "Haomin Wen, Youfang Lin, Yutong Xia, Huaiyu Wan, Qingsong Wen, Roger Zimmermann, Yuxuan Liang",
        "year": 2023,
        "category": "Generative Diffusion Models for Spatiotemporal Graphs"
    },
    {
        "id": "2606.01634",
        "filename": "06_E4GEN_Extreme_Enhanced_Timeseries_Generation_Diffusion.pdf",
        "title": "E4GEN: Event-level Explainable Extreme-Enhanced Time-series Generation",
        "authors": "Zepu Wang, et al.",
        "year": 2026,
        "category": "Rare/Extreme-Event Generative Augmentation with Diffusion"
    },
    {
        "id": "2510.20651",
        "filename": "07_xTime_Extreme_Event_Prediction_Distillation_MoE.pdf",
        "title": "xTime: Extreme Event Prediction with Hierarchical Knowledge Distillation and Expert Fusion",
        "authors": "Jianyu Chen, et al.",
        "year": 2025,
        "category": "Rare Event Forecasting & Mixture of Experts (MoE)"
    },
    {
        "id": "2107.03502",
        "filename": "08_CSDI_Conditional_Score_Diffusion_Imputation_Forecasting.pdf",
        "title": "CSDI: Conditional Score-based Diffusion Models for Probabilistic Imputation and Forecasting",
        "authors": "Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon (NeurIPS 2021)",
        "year": 2021,
        "category": "Foundational Time Series Diffusion Model"
    },
    {
        "id": "2101.12072",
        "filename": "09_TimeGrad_Autoregressive_Diffusion_Multivariate_Forecasting.pdf",
        "title": "TimeGrad: Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting",
        "authors": "Rasul Kashif, Calvin Seward, et al. (ICML 2021)",
        "year": 2021,
        "category": "Foundational Autoregressive Diffusion"
    },
    {
        "id": "2404.18886",
        "filename": "10_Survey_Diffusion_Models_Timeseries_Spatiotemporal.pdf",
        "title": "A Survey on Diffusion Models for Time Series and Spatio-Temporal Data",
        "authors": "Yiyuan Yang, et al.",
        "year": 2024,
        "category": "Comprehensive Review of Generative Diffusion in Spatiotemporal"
    },
    {
        "id": "2205.13947",
        "filename": "11_ST_GFSL_Spatiotemporal_Graph_FewShot_CrossCity.pdf",
        "title": "ST-GFSL: Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer",
        "authors": "Bin Lu, Xiaoying Gan, Weinan Zhang, et al. (KDD 2022)",
        "year": 2022,
        "category": "Cross-City Few-Shot Transfer Learning & Meta-GNNs"
    },
    {
        "id": "2303.07184",
        "filename": "12_TrafficTL_Transfer_Learning_Mutual_Information.pdf",
        "title": "Traffic Prediction with Transfer Learning: A Mutual Information-based Approach",
        "authors": "Linfeng Huang, et al.",
        "year": 2023,
        "category": "Transfer Learning & Negative Transfer Mitigation"
    },
    {
        "id": "2308.09727",
        "filename": "13_TPB_CrossCity_FewShot_Traffic_Pattern_Bank.pdf",
        "title": "Cross-city Few-Shot Traffic Forecasting via Traffic Pattern Bank",
        "authors": "Zhanyu Liu, Guanjie Zheng, Yanwei Yu (CIKM 2023)",
        "year": 2023,
        "category": "Pattern Bank / Cross-Domain Spatial Transfer"
    },
    {
        "id": "2412.00316",
        "filename": "14_HiMoE_Heterogeneity_Informed_Mixture_of_Experts.pdf",
        "title": "HiMoE: Heterogeneity-Informed Mixture-of-Experts for Fair Spatial-Temporal Forecasting",
        "authors": "Shaohan Yu, Pan Deng, Yu Zhao, et al.",
        "year": 2024,
        "category": "Mixture-of-Experts (MoE) for Spatio-Temporal Heterogeneity"
    },
    {
        "id": "2410.14099",
        "filename": "15_ST_MoE_BERT_Spatiotemporal_MoE_CrossCity.pdf",
        "title": "ST-MoE-BERT: A Spatial-Temporal Mixture-of-Experts Framework for Long-Term Cross-City Mobility Prediction",
        "authors": "Haoyu He, Haozheng Luo, Qi R. Wang (ACM SIGSPATIAL 2024)",
        "year": 2024,
        "category": "Cross-City Mixture-of-Experts & Foundation Transformer"
    },
    {
        "id": "2509.25449",
        "filename": "16_TS_JEPA_Joint_Embeddings_Go_Temporal_SelfSupervised.pdf",
        "title": "Joint Embeddings Go Temporal (TS-JEPA: Self-Supervised Time Series Representation)",
        "authors": "Jean-Yves Franceschi, et al. (NeurIPS Workshop)",
        "year": 2025,
        "category": "JEPA (Joint Embedding Predictive Architecture) for Time Series"
    },
    {
        "id": "2403.00813",
        "filename": "17_UrbanGPT_Spatiotemporal_Large_Language_Models.pdf",
        "title": "UrbanGPT: Spatio-Temporal Large Language Models",
        "authors": "Zhonghang Li, et al. (KDD 2024)",
        "year": 2024,
        "category": "Spatio-Temporal LLMs & Cross-City Zero-Shot Transfer"
    },
    {
        "id": "2511.00096",
        "filename": "18_Urban_MAS_LLM_MultiAgent_Urban_Prediction.pdf",
        "title": "Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System",
        "authors": "Shangyu Lou (ACM SIGSPATIAL UrbanAI)",
        "year": 2025,
        "category": "Multi-Agent System (MAS / A2A) for Urban Prediction"
    },
    {
        "id": "2211.14730",
        "filename": "19_PatchTST_A_Time_Series_Is_Worth_64_Words.pdf",
        "title": "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)",
        "authors": "Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam (ICLR 2023)",
        "year": 2023,
        "category": "State-of-the-Art Transformer Architecture (Patching)"
    },
    {
        "id": "2210.02186",
        "filename": "20_TimesNet_Temporal_2D_Variation_Modeling.pdf",
        "title": "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis",
        "authors": "Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long (ICLR 2023)",
        "year": 2023,
        "category": "Temporal 2D Variation & Multi-Periodicity Modeling"
    },
    {
        "id": "2310.06625",
        "filename": "21_iTransformer_Inverted_Transformers_Effective_Forecasting.pdf",
        "title": "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
        "authors": "Yong Liu, Haixu Wu, Jianmin Wang, Mingsheng Long (ICLR 2024 Spotlight)",
        "year": 2024,
        "category": "Inverted Multi-Variate Transformer Architecture"
    },
    {
        "id": "2210.02410",
        "filename": "22_Crossformer_Cross_Dimension_Multivariate_Forecasting.pdf",
        "title": "Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting",
        "authors": "Yunhao Zhang, Junchi Yan (ICLR 2023)",
        "year": 2023,
        "category": "Cross-Dimension Dependency Transformer"
    },
    {
        "id": "2310.01728",
        "filename": "23_Time_LLM_Reprogramming_LLMs_for_Time_Series.pdf",
        "title": "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
        "authors": "Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y. Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Shirui Pan, Qingsong Wen (ICLR 2024)",
        "year": 2024,
        "category": "LLM Reprogramming for Time Series Forecasting"
    },
    {
        "id": "2403.07815",
        "filename": "24_Chronos_Learning_the_Language_of_Time_Series.pdf",
        "title": "Chronos: Learning the Language of Time Series",
        "authors": "Abdul Fatir Ansari, et al. (Amazon Research)",
        "year": 2024,
        "category": "Pretrained Time Series Foundation Models"
    },
    {
        "id": "2403.00131",
        "filename": "25_UniTS_Unified_Multi_Task_Time_Series_Model.pdf",
        "title": "UniTS: Building a Unified Time Series Model",
        "authors": "Shanghua Gao, et al. (NeurIPS 2024)",
        "year": 2024,
        "category": "Unified Multi-Task Time Series Foundation Model"
    },
    {
        "id": "2310.10196",
        "filename": "26_Large_Models_Timeseries_Spatiotemporal_Survey.pdf",
        "title": "Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook",
        "authors": "Ming Jin, Qingsong Wen, Yuxuan Liang, Chaoli Zhang, Shirui Pan, Philip S. Yu",
        "year": 2023,
        "category": "Comprehensive Foundation Models Survey for Spatiotemporal"
    },
    {
        "id": "2402.01801",
        "filename": "27_LLMs_for_Time_Series_Survey_Taxonomy.pdf",
        "title": "Large Language Models for Time Series: A Survey",
        "authors": "Xiyuan Zhang, Ranak Roy Chowdhury, Rajesh K. Gupta, Jingbo Shang (IJCAI 2024)",
        "year": 2024,
        "category": "LLMs for Time Series Methodological Taxonomy"
    },
    {
        "id": "2102.10551",
        "filename": "28_Delhi_AQI_Prediction_LSTM_Lockdown.pdf",
        "title": "Delhi Air Quality Prediction Using LSTM Deep Learning Models with Focus on Lockdown Anomalies",
        "authors": "Arun Sharma, et al.",
        "year": 2021,
        "category": "Indian AQI Modeling, Delhi Ground Truth & Abrupt Shift Handling"
    },
    {
        "id": "2404.08702",
        "filename": "29_India_AQI_Stubble_Burning_Variability_Forecasting.pdf",
        "title": "Predictive Modelling of Air Quality Index (AQI) Across Diverse Indian Cities: Stubble Burning Impact",
        "authors": "Rahul Kumar, et al.",
        "year": 2024,
        "category": "Multi-City Indian AQI, Cross-State Agricultural Extreme Spikes"
    },
    {
        "id": "2112.05753",
        "filename": "30_New_Delhi_AQI_Machine_Learning_Evaluation.pdf",
        "title": "Using Machine Learning to Predict Air Quality Index in New Delhi",
        "authors": "Anshul Verma, et al.",
        "year": 2021,
        "category": "CPCB & US Embassy Delhi Benchmark Pollutant Modeling"
    }
]

out_dir = os.path.join(os.getcwd(), 'papers')
os.makedirs(out_dir, exist_ok=True)

# Save paper metadata catalog as JSON
meta_file = os.path.join(out_dir, 'papers_metadata.json')
with open(meta_file, 'w', encoding='utf-8') as f:
    json.dump(papers, f, indent=2)

print(f"Metadata saved. Starting downloads for {len(papers)} papers into '{out_dir}'...")

success = 0
failed = []

for i, p in enumerate(papers, 1):
    url = f"https://arxiv.org/pdf/{p['id']}.pdf"
    filepath = os.path.join(out_dir, p['filename'])
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 20000:
        size_kb = os.path.getsize(filepath) / 1024
        print(f"[{i}/{len(papers)}] Already exists ({size_kb:.1f} KB): {p['filename']}")
        success += 1
        continue
    
    print(f"[{i}/{len(papers)}] Downloading {p['id']} -> {p['filename']}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
            if data.startswith(b'%PDF'):
                with open(filepath, 'wb') as f:
                    f.write(data)
                size_kb = len(data) / 1024
                print(f"   -> OK! Saved ({size_kb:.1f} KB)")
                success += 1
            else:
                print(f"   -> ERROR: Header is not PDF: {data[:20]}")
                failed.append((p['id'], "Invalid Header"))
    except Exception as e:
        print(f"   -> ERROR: {e}")
        failed.append((p['id'], str(e)))
    time.sleep(1.2)

print(f"\nCompleted: {success}/{len(papers)} papers successfully stored in {out_dir}.")
if failed:
    print("Failures encountered:", failed)
