# Project: Open-ended Text Generation

[![My Skills](https://skillicons.dev/icons?i=py)](https://www.python.org/)
[![My Skills](https://skillicons.dev/icons?i=tensorflow)](https://www.tensorflow.org/?hl=fr)
[![My Skills](https://skillicons.dev/icons?i=github)](https://github.com/RaykeshR/PFE-Roguelike)
[![My Skills](https://skillicons.dev/icons?i=git)](https://git-scm.com/)
[![My Skills](https://skillicons.dev/icons?i=bash)](https://fr.wikibooks.org/wiki/Programmation_Bash/Scripts)
[![My Skills](https://skillicons.dev/icons?i=md)](https://docs.github.com/fr/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
[![My Skills](https://skillicons.dev/icons?i=vscode)](https://code.visualstudio.com/)
[![My Skills](https://skillicons.dev/icons?i=powershell)](https://learn.microsoft.com/fr-fr/powershell/scripting/overview?view=powershell-7.4)
[![My Skills](https://skillicons.dev/icons?i=windows)](https://www.microsoft.com/fr-fr/windows?r=1)

<!-- Le site est accessible via ce [Lien qui fait une Redirection d'URL](https://raykeshr.github.io/PFE-Roguelike/) vers une page d'accueil pour le site du Github : PFE-Roguelike -->
Rapport : [Fichier Word](https://reseaueseo-my.sharepoint.com/:w:/r/personal/lea_ludet_reseau_eseo_fr/Documents/Rapport_Projet_NLP.docx?d=w0ac3619c8d6b469e8e2f4897154e7524&csf=1&web=1&e=r4mTnR) 


#### Sommaire 

TODO :octocat: :neckbeard: :bowtie: :shipit:

## Résultat de l'analyse/évaluation : 

============================================================================================================================================
| Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Method | Parameters | MAUVE | Gen_Length | rep-2 | rep-3 | rep-4 | Coh_gpt2 | Coh_gpt2-large | Coh_gpt2-medium | Coh_gpt2-xl | Coh_opt-1.3b | Coh_opt-125m | Coh_opt-2.7b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.4}$ | $\textsf{3.4200}$ | $\textsf{90.3600}$ | $\textsf{0.8533}$ | $\textsf{0.8090}$ | $\textsf{0.7745}$ | $\textsf{-1.1207}$ | $\textsf{-1.6835}$ | $\textsf{-1.1248}$ | $\textsf{-1.4209}$ | $\textsf{-0.9934}$ | $\textsf{-1.1265}$ | $\textsf{-0.9605}$ |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.6}$ | $\textsf{1.4400}$ | $\textsf{60.0000}$ | $\textsf{0.7459}$ | $\textsf{0.6839}$ | $\textsf{0.6413}$ | $\textsf{-1.1724}$ | $\textsf{-2.0421}$ | $\textsf{-1.1640}$ | $\textsf{-1.6741}$ | $\textsf{-1.0013}$ | $\textsf{-1.1750}$ | $\textsf{-0.9624}$ |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.8}$ | $\textsf{0.7300}$ | $\textsf{27.9400}$ | $\textsf{0.5571}$ | $\textsf{0.4497}$ | $\textsf{0.3753}$ | $\textsf{-1.2079}$ | $\textsf{-2.4594}$ | $\textsf{-1.2093}$ | $\textsf{-1.9477}$ | $\textsf{-1.0027}$ | $\textsf{-1.2069}$ | $\textsf{-0.9582}$ |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.4}$ | $\textsf{5.4600}$ | $\textsf{90.6200}$ | $\textsf{0.8682}$ | $\textsf{0.8268}$ | $\textsf{0.7954}$ | $\textsf{-1.0036}$ | $\textsf{-1.5675}$ | $\textsf{-1.0137}$ | $\textsf{-1.2978}$ | $\textsf{-0.8858}$ | $\textsf{-1.0346}$ | $\textsf{-0.8588}$ |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.6}$ | $\textsf{1.1900}$ | $\textsf{71.0900}$ | $\textsf{0.8159}$ | $\textsf{0.7629}$ | $\textsf{0.7170}$ | $\textsf{-1.0707}$ | $\textsf{-1.8471}$ | $\textsf{-1.0742}$ | $\textsf{-1.4810}$ | $\textsf{-0.9282}$ | $\textsf{-1.0877}$ | $\textsf{-0.8988}$ |
| $\textsf{gpt2}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.8}$ | $\textsf{1.2800}$ | $\textsf{45.5100}$ | $\textsf{0.7165}$ | $\textsf{0.6293}$ | $\textsf{0.5512}$ | $\textsf{-1.1255}$ | $\textsf{-2.1786}$ | $\textsf{-1.1388}$ | $\textsf{-1.7167}$ | $\textsf{-0.9486}$ | $\textsf{-1.1303}$ | $\textsf{-0.9136}$ |
| $\textsf{gpt2}$ | $\textsf{Greedy}$ | $\textsf{N/A}$ | $\textsf{2.2900}$ | $\color{red}{\textsf{202.2100}}$ | $\textsf{0.8519}$ | $\textsf{0.8349}$ | $\textsf{0.8215}$ | $\color{red}{\textsf{-0.2882}}$ | $\color{red}{\textsf{-0.3184}}$ | $\color{red}{\textsf{-0.3393}}$ | $\color{red}{\textsf{-0.3353}}$ | $\color{red}{\textsf{-0.3126}}$ | $\color{red}{\textsf{-0.3192}}$ | $\color{red}{\textsf{-0.3244}}$ |
| $\textsf{gpt2}$ | $\textsf{Nucleus}$ | $\textsf{p=0.95}$ | $\textsf{34.7600}$ | $\textsf{189.5300}$ | $\textsf{0.1092}$ | $\textsf{0.0430}$ | $\textsf{0.0251}$ | $\textsf{-2.4129}$ | $\textsf{-2.7282}$ | $\textsf{-2.7004}$ | $\textsf{-2.8097}$ | $\textsf{-2.7949}$ | $\textsf{-2.6613}$ | $\textsf{-2.8488}$ |
| $\color{blue}{\textsf{gpt2}}$ | $\color{blue}{\textsf{Typical}}$ | $\color{blue}{\textsf{p=0.95}}$ | $\color{blue}{\textsf{43.0300}}$ | $\color{blue}{\textsf{186.4100}}$ | $\color{blue}{\textsf{0.1138}}$ | $\color{blue}{\textsf{0.0472}}$ | $\color{blue}{\textsf{0.0283}}$ | $\color{blue}{\textsf{-2.3452}}$ | $\color{blue}{\textsf{-2.6646}}$ | $\color{blue}{\textsf{-2.6464}}$ | $\color{blue}{\textsf{-2.7366}}$ | $\color{blue}{\textsf{-2.7316}}$ | $\color{blue}{\textsf{-2.6207}}$ | $\color{blue}{\textsf{-2.7927}}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.4}$ | $\textsf{2.3600}$ | $\textsf{189.0800}$ | $\textsf{0.9708}$ | $\textsf{0.9676}$ | $\textsf{0.9647}$ | $\textsf{-0.6488}$ | $\textsf{-0.7385}$ | $\textsf{-0.6842}$ | $\textsf{-0.7371}$ | $\textsf{-0.6042}$ | $\textsf{-0.7388}$ | $\textsf{-0.5656}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.6}$ | $\textsf{3.6000}$ | $\textsf{188.9600}$ | $\textsf{0.9546}$ | $\textsf{0.9515}$ | $\textsf{0.9487}$ | $\textsf{-0.7825}$ | $\textsf{-0.8729}$ | $\textsf{-0.8137}$ | $\textsf{-0.8556}$ | $\textsf{-0.7604}$ | $\textsf{-0.8439}$ | $\textsf{-0.7069}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.8}$ | $\textsf{6.9100}$ | $\textsf{189.1900}$ | $\textsf{0.9121}$ | $\textsf{0.8972}$ | $\textsf{0.8840}$ | $\textsf{-1.0900}$ | $\textsf{-1.1803}$ | $\textsf{-1.1144}$ | $\textsf{-1.1585}$ | $\textsf{-1.0978}$ | $\textsf{-1.1620}$ | $\textsf{-1.0279}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.4}$ | $\textsf{2.8700}$ | $\textsf{189.7900}$ | $\textsf{0.9780}$ | $\textsf{0.9754}$ | $\textsf{0.9728}$ | $\textsf{-0.6090}$ | $\textsf{-0.7141}$ | $\textsf{-0.6508}$ | $\textsf{-0.6746}$ | $\textsf{-0.5592}$ | $\textsf{-0.7036}$ | $\textsf{-0.5281}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.6}$ | $\textsf{4.1000}$ | $\textsf{188.9100}$ | $\textsf{0.9686}$ | $\textsf{0.9666}$ | $\textsf{0.9648}$ | $\textsf{-0.6502}$ | $\textsf{-0.7452}$ | $\textsf{-0.6863}$ | $\textsf{-0.7246}$ | $\textsf{-0.6259}$ | $\textsf{-0.7281}$ | $\textsf{-0.5894}$ |
| $\textsf{gpt2-large}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.8}$ | $\textsf{2.7800}$ | $\textsf{189.3400}$ | $\textsf{0.9483}$ | $\textsf{0.9383}$ | $\textsf{0.9304}$ | $\textsf{-0.8139}$ | $\textsf{-0.8748}$ | $\textsf{-0.8343}$ | $\textsf{-0.8445}$ | $\textsf{-0.7959}$ | $\textsf{-0.8881}$ | $\textsf{-0.7436}$ |
| $\textsf{gpt2-large}$ | $\textsf{Greedy}$ | $\textsf{N/A}$ | $\textsf{4.1900}$ | $\textsf{196.7200}$ | $\textsf{0.7823}$ | $\textsf{0.7549}$ | $\textsf{0.7366}$ | $\textsf{-0.5233}$ | $\textsf{-0.3784}$ | $\textsf{-0.4736}$ | $\textsf{-0.4237}$ | $\textsf{-0.4219}$ | $\textsf{-0.5082}$ | $\textsf{-0.4303}$ |
| $\color{blue}{\textsf{gpt2-large}}$ | $\color{blue}{\textsf{Nucleus}}$ | $\color{blue}{\textsf{p=0.95}}$ | $\color{blue}{\textsf{49.3200}}$ | $\color{blue}{\textsf{178.0200}}$ | $\color{blue}{\textsf{0.1232}}$ | $\color{blue}{\textsf{0.0582}}$ | $\color{blue}{\textsf{0.0403}}$ | $\color{blue}{\textsf{-2.4388}}$ | $\color{blue}{\textsf{-1.9818}}$ | $\color{blue}{\textsf{-2.2595}}$ | $\color{blue}{\textsf{-2.1908}}$ | $\color{blue}{\textsf{-2.2350}}$ | $\color{blue}{\textsf{-2.5068}}$ | $\color{blue}{\textsf{-2.2520}}$ |
| $\textsf{gpt2-large}$ | $\textsf{Typical}$ | $\textsf{p=0.95}$ | $\textsf{38.7300}$ | $\textsf{177.0900}$ | $\textsf{0.1099}$ | $\textsf{0.0458}$ | $\textsf{0.0297}$ | $\textsf{-2.4777}$ | $\textsf{-2.0221}$ | $\textsf{-2.2933}$ | $\textsf{-2.2131}$ | $\textsf{-2.2732}$ | $\textsf{-2.5692}$ | $\textsf{-2.2969}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.4}$ | $\textsf{0.5700}$ | $\textsf{7.6000}$ | $\textsf{0.8389}$ | $\textsf{0.8705}$ | $\textsf{0.8892}$ | $\textsf{-0.7574}$ | $\textsf{-1.9972}$ | $\textsf{-0.7951}$ | $\textsf{-1.2896}$ | $\textsf{-0.5582}$ | $\textsf{-0.7654}$ | $\textsf{-0.5176}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.6}$ | $\textsf{0.4300}$ | $\textsf{3.0300}$ | $\textsf{0.6667}$ | $\textsf{0.7628}$ | $\textsf{0.8235}$ | $\textsf{-0.7710}$ | $\textsf{-2.0161}$ | $\textsf{-0.8122}$ | $\textsf{-1.3115}$ | $\textsf{-0.5664}$ | $\textsf{-0.7814}$ | $\textsf{-0.5275}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.8}$ | $\textsf{0.6400}$ | $\textsf{2.1200}$ | $\textsf{0.5135}$ | $\textsf{0.6418}$ | $\textsf{0.7344}$ | $\textsf{-0.7648}$ | $\textsf{-2.0037}$ | $\textsf{-0.8040}$ | $\textsf{-1.2945}$ | $\textsf{-0.5659}$ | $\textsf{-0.7791}$ | $\textsf{-0.5256}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.4}$ | $\textsf{0.6400}$ | $\textsf{8.7300}$ | $\textsf{0.7828}$ | $\textsf{0.8184}$ | $\textsf{0.8469}$ | $\textsf{-0.8337}$ | $\textsf{-2.0271}$ | $\textsf{-0.8632}$ | $\textsf{-1.3272}$ | $\textsf{-0.6256}$ | $\textsf{-0.8376}$ | $\textsf{-0.5873}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.6}$ | $\textsf{0.4600}$ | $\textsf{4.2100}$ | $\textsf{0.4438}$ | $\textsf{0.5033}$ | $\textsf{0.5552}$ | $\textsf{-0.8536}$ | $\textsf{-2.0994}$ | $\textsf{-0.8846}$ | $\textsf{-1.3672}$ | $\textsf{-0.6413}$ | $\textsf{-0.8649}$ | $\textsf{-0.6106}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.8}$ | $\textsf{0.4500}$ | $\textsf{3.3500}$ | $\textsf{0.2720}$ | $\textsf{0.3450}$ | $\textsf{0.4028}$ | $\textsf{-0.8519}$ | $\textsf{-2.1414}$ | $\textsf{-0.8979}$ | $\textsf{-1.3862}$ | $\textsf{-0.6460}$ | $\textsf{-0.8669}$ | $\textsf{-0.6110}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Greedy}$ | $\textsf{N/A}$ | $\textsf{4.7900}$ | $\textsf{200.8800}$ | $\textsf{0.8095}$ | $\textsf{0.7816}$ | $\textsf{0.7619}$ | $\textsf{-0.4620}$ | $\textsf{-0.3970}$ | $\textsf{-0.3899}$ | $\textsf{-0.4167}$ | $\textsf{-0.3996}$ | $\textsf{-0.4584}$ | $\textsf{-0.4142}$ |
| $\color{green}{\textsf{gpt2-medium}}$ | $\color{green}{\textsf{Nucleus}}$ | $\color{green}{\textsf{p=0.95}}$ | $\color{red}{\textsf{56.1200}}$ | $\color{green}{\textsf{183.7400}}$ | $\color{green}{\textsf{0.0876}}$ | $\color{green}{\textsf{0.0310}}$ | $\color{green}{\textsf{0.0178}}$ | $\color{green}{\textsf{-2.6471}}$ | $\color{green}{\textsf{-2.4792}}$ | $\color{green}{\textsf{-2.3126}}$ | $\color{green}{\textsf{-2.5317}}$ | $\color{green}{\textsf{-2.5636}}$ | $\color{green}{\textsf{-2.7203}}$ | $\color{green}{\textsf{-2.6041}}$ |
| $\textsf{gpt2-medium}$ | $\textsf{Typical}$ | $\textsf{p=0.95}$ | $\textsf{44.2800}$ | $\textsf{186.6500}$ | $\textsf{0.1128}$ | $\textsf{0.0539}$ | $\textsf{0.0401}$ | $\textsf{-2.5455}$ | $\textsf{-2.3962}$ | $\textsf{-2.2376}$ | $\textsf{-2.4516}$ | $\textsf{-2.4820}$ | $\textsf{-2.6267}$ | $\textsf{-2.5244}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.4}$ | $\textsf{4.4300}$ | $\textsf{115.2800}$ | $\textsf{0.0169}$ | $\textsf{0.0119}$ | $\textsf{0.0111}$ | $\textsf{-9.1670}$ | $\textsf{-9.0882}$ | $\textsf{-9.0463}$ | $\textsf{-8.8964}$ | $\textsf{-8.8832}$ | $\textsf{-9.1991}$ | $\textsf{-8.8201}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.6}$ | $\textsf{4.5100}$ | $\textsf{112.6900}$ | $\textsf{0.0030}$ | $\textsf{0.0010}$ | $\textsf{0.0007}$ | $\textsf{-9.2562}$ | $\textsf{-9.1617}$ | $\textsf{-9.1262}$ | $\textsf{-8.9895}$ | $\textsf{-8.9678}$ | $\textsf{-9.2803}$ | $\textsf{-8.8993}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=10, a=0.8}$ | $\textsf{4.1100}$ | $\textsf{113.2100}$ | $\color{red}{\textsf{0.0015}}$ | $\color{red}{\textsf{0.0000}}$ | $\color{red}{\textsf{0.0000}}$ | $\textsf{-9.2842}$ | $\textsf{-9.1628}$ | $\textsf{-9.1348}$ | $\textsf{-8.9875}$ | $\textsf{-8.9932}$ | $\textsf{-9.3025}$ | $\textsf{-8.9280}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.4}$ | $\textsf{11.9400}$ | $\textsf{120.3100}$ | $\textsf{0.1386}$ | $\textsf{0.1170}$ | $\textsf{0.1097}$ | $\textsf{-8.3395}$ | $\textsf{-8.3691}$ | $\textsf{-8.3028}$ | $\textsf{-8.0903}$ | $\textsf{-7.9099}$ | $\textsf{-8.3573}$ | $\textsf{-7.8405}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.6}$ | $\textsf{12.3300}$ | $\textsf{116.2800}$ | $\textsf{0.0709}$ | $\textsf{0.0508}$ | $\textsf{0.0442}$ | $\textsf{-8.7235}$ | $\textsf{-8.6956}$ | $\textsf{-8.6602}$ | $\textsf{-8.4190}$ | $\textsf{-8.2650}$ | $\textsf{-8.7586}$ | $\textsf{-8.2096}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Contrastive}$ | $\textsf{k=5, a=0.8}$ | $\textsf{15.0600}$ | $\textsf{114.5700}$ | $\textsf{0.0355}$ | $\textsf{0.0158}$ | $\textsf{0.0112}$ | $\textsf{-8.8671}$ | $\textsf{-8.8114}$ | $\textsf{-8.7565}$ | $\textsf{-8.5626}$ | $\textsf{-8.4016}$ | $\textsf{-8.9393}$ | $\textsf{-8.3589}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Greedy}$ | $\textsf{N/A}$ | $\textsf{5.0200}$ | $\textsf{196.5400}$ | $\textsf{0.7559}$ | $\textsf{0.7182}$ | $\textsf{0.6913}$ | $\textsf{-0.5588}$ | $\textsf{-0.4263}$ | $\textsf{-0.4982}$ | $\textsf{-0.3774}$ | $\textsf{-0.4278}$ | $\textsf{-0.5575}$ | $\textsf{-0.4267}$ |
| $\color{blue}{\textsf{gpt2-xl}}$ | $\color{blue}{\textsf{Nucleus}}$ | $\color{blue}{\textsf{p=0.95}}$ | $\color{blue}{\textsf{50.9000}}$ | $\color{blue}{\textsf{178.9200}}$ | $\color{blue}{\textsf{0.1125}}$ | $\color{blue}{\textsf{0.0475}}$ | $\color{blue}{\textsf{0.0296}}$ | $\color{blue}{\textsf{-2.5551}}$ | $\color{blue}{\textsf{-2.1907}}$ | $\color{blue}{\textsf{-2.3259}}$ | $\color{blue}{\textsf{-1.9738}}$ | $\color{blue}{\textsf{-2.2540}}$ | $\color{blue}{\textsf{-2.6280}}$ | $\color{blue}{\textsf{-2.2513}}$ |
| $\textsf{gpt2-xl}$ | $\textsf{Typical}$ | $\textsf{p=0.95}$ | $\textsf{38.7700}$ | $\textsf{183.0500}$ | $\textsf{0.1208}$ | $\textsf{0.0552}$ | $\textsf{0.0386}$ | $\textsf{-2.5367}$ | $\textsf{-2.1636}$ | $\textsf{-2.3012}$ | $\textsf{-1.9591}$ | $\textsf{-2.2234}$ | $\textsf{-2.6208}$ | $\textsf{-2.2264}$ |

============================================================================================================================================

## Generation

```py
python -m open_text_gen.generate --alphas 0.2 0.5 0.8 --dataset_name wikitext --output_dir open_text_gen/wikitext
python -m open_text_gen.generate --alphas 0.2 0.5 0.8 --dataset_name wikitext --output_dir open_text_gen/wikitext --num_prefixes 5
```

## Evaluation

### Coherence

```bash
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
python open_text_gen/compute_coherence.py --opt_model_name facebook/opt-2.7b --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```

### Diversity, MAUVE, and Generation Length

```bash
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.2_gpt2-xl_256.jsonl
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.5_gpt2-xl_256.jsonl
python open_text_gen/measure_diversity_mauve_gen_length.py --test_path open_text_gen/wikitext/wikitext_contrastive-alpha-0.8_gpt2-xl_256.jsonl
```

### run_experiment.py
```bash
python run_experiment.py --model_name gpt2 --dataset_name wikitext --num_prefixes 5 --alphas 0.6 --decoding_len 50
python grid_search.py
pip install transformers==4.35.2 ?
```

Pour tester des modèles de Ollama directement (texte généré par Ollama puis évalué)
```bash
python open_text_gen/generate_ollama.py --models llama3 mistral --num_prefixes 100
python run_eval_ollama.py
```

Téléchargement des Données/ Model
```CMD
%USERPROFILE%\.cache\huggingface
```

Pour utiliser Cuda/GPU : 
```bash
https://developer.nvidia.com/cuda-13-0-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip uninstall torch torchvision torchaudio -y
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
nvidia-smi

```
(les dernières lignes sont là en cas de bug)

### Mise en place (Windows):

<!-- <details open> -->
<details>
<summary>Création d'environnements virtuels : </summary>

## Un package manquant : 
<!-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -->
```
.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt
```
## Version 1 ligne/rapide : 
```py
python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt
```
(en cas de bug faire les étapes ci-dessous ou essayer sur CMD ou powershell>=7.0)
```$env:SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL='True'; ./.venv/Scripts/python.exe -m pip install -r requirements.txt```
```set SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True```

### 1. Cloner le Repo

avec GitHub (Copie les fichiers localement)

### 2. `python -m venv .venv`

peut nécessiter le passage par CMD (Crée le Dossier .venv)

### 3. `.venv\Scripts\activate`
 

Créer un environnement virtuel Python (Sur Linux/Mac) :
```bash
source venv/bin/activate  # Sur Linux/Mac
```

Lancer avec le CMD peut éviter les erreurs. (Lance l'environnement virtuel)
EN ADMIN : `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned` en cas d'erreur
([détail](https://tutorial.djangogirls.org/fr/django_installation/))

Résultat : 

$\color{rgba(100,255,100, 0.75)}{\textsf{(.venv)}}$ PS C:\Users...\Portfolio_Django> |

On peut aussi (Si c'est un problème de l'éditeur) `$ . .venv\Scripts\activate.ps1`
(lance l'environnement virtuel)

### 4. `python -m pip install --upgrade pip`

(met à jour pip)

### 5. `python -m pip install -r requirements.txt`

```pip freeze > requirements.txt``` pour remplir automatiquement les requirements

Pour toutes les étapes précédentes (sur CMD ou powershell>=7) : 


```
python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt
```

en cas d'erreur (supprimer le dossier .venv ou lancer): 

```.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt```

Avec  pip freeze  :

    Pour toutes les étapes précédentes (sur CMD ou powershell>=7) : 
    ```python -m venv .venv && .venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt```
    
    en cas d'erreur (supprimer le dossier .venv ou lancer): 
    ```.venv\Scripts\activate && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip freeze > requirements.txt```
    
### 6. Modifier .git\info\exclude 

Ajouter : `.venv`
(Ne prend pas en compte la modification du dossier .venv)

### 7. Lancer le fichier main.py 

Commande : `python main.py `
(Lance le fichier principal avec python)

### Linux/Mac :

"""
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
"""
Modifier .git\info\exclude 

</details> 


<details>
<summary>Nettoyer un dépôt git : </summary>
Télécharger BFG Repo-Cleaner sur le site (.jar): 
https://rtyley.github.io/bfg-repo-cleaner/

Lancer : 

git clone --mirror https://github.com/RaykeshR/PFE-Roguelike.git
cd PFE-Roguelike.git
<!-- java -jar ../bfg-1.15.0.jar --delete-files database/.env -->
java -jar ../bfg-1.15.0.jar --delete-files .env

git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force --all
git push --force --tags

<!-- git push --force origin Dev -->

<!-- git push --force origin --all
git push --force origin --tags -->

git push --mirror

</details> 

<br>

<details>
<summary>Autre : </summary>
<details>
<summary>Gemini-cli : </summary>
1. ouvrir un terminal (WSL, ...)
2. taper : `npm install -g @google/gemini-cli` / `sudo npm install -g @google/gemini-cli`
3. Changer de dossier : `cd .../PFE-Roguelike`
4. lancer gemini : avec `gemini`
5. login avec google
6. tester avec une question
7. lancer la commande : `/init`
    
</details> 
<details>
<summary>Gemini-cli + MCP Github : </summary>

Le Model Context Protocol (MCP) est un protocole standard ouvert conçu pour connecter des modèles d'intelligence artificielle (IA) (LLM, ...)
Ici, le MCP Github permettra à gemini d'accéder aux code source sans passer par une recherche web à chaque fois.

1. ouvrir un terminal (WSL, ...)
2. taper : `cd ~/.gemini`                ( %USERPROFILE%\.gemini )
3. modifier le fichier settings.json et ajouter au json : 
```
    , 
    "mcpServers": {
        "github": {
            "httpUrl": "https://api.githubcopilot.com/mcp/",
            "headers": {
                    "Authorization": "Bearer ghp_..."
                },
                "timeout": 5000
        }
    }

```    

> [!NOTE]  
> Pour obtenir le "Bearer ghp_..." ou plus précisément le `ghp_...` il faut mettre un PAT ( Personal access tokens (classic) : [https://github.com/settings/tokens](https://github.com/settings/tokens) ) nommé de préférence "Gemini MCP" avec les droits voulus.

</details> 

</details> 