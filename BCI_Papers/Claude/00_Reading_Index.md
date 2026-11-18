# Closed-Loop Affective Stimulus Induction — Reading Index
Research Methods theme for Matei's CPS master's work: methods for inducing affective states, plus how to do it inside a closed loop.
Linked to ANNEX 1, Task 3 (interleaved four-quadrant elicitation) and Task 6 (MPC/RL controller driving music, lighting and haptics).
All PDFs are legal open-access copies (publisher OA, arXiv, NSF-PAR, author or university repositories). Compiled 6 Oct 2026.

## 01_Cornerstone (foundational, pre-2021)
| File | Why it matters for your project |
|---|---|
| 1980_Russell_Circumplex_Model_of_Affect | The valence-arousal space your set-point lives in (ref [2] in your proposal). |
| 1995_Gross_Levenson_Emotion_Elicitation_Using_Films | The classic validated film set and the criteria for a "good" induction stimulus (intensity and discreteness). |
| 2012_Koelstra_DEAP_Database | Music-video induction plus SAM ratings; your bachelor benchmark (ref [3]). |
| 2014_Zhang_Barrett_Comparison_Four_Affect_Induction_Procedures | Head-to-head comparison of four induction procedures. Useful for justifying your modality choice. |
| 2015_Ramirez_Musical_Neurofeedback_Depression | Early EEG valence/arousal to music-parameter loop (tempo, loudness). |
| 2016_Daly_Affective_Brain_Computer_Music_Interfacing | Affective BCMI architecture: EEG-decoded affect drives generated music. |
| 2016_Ewing_Fairclough_Adaptive_Game_Biocybernetic_Loop | Biocybernetic-loop method: validate the EEG index first, then close the loop. |
| 2016_Lorenz_Automatic_Neuroscientist_ClosedLoop_Design | Bayesian optimisation picks the next stimulus in real time, a template for adaptive stimulus selection. |
| 2017_Azevedo_Haptic_Heartbeat_Wearable_Calming_Effect | Slow heartbeat-like vibration lowers arousal, evidence that haptics can serve as an actuator. |
| 2018_MarinMorales_Affective_Computing_in_VR_EEG_HRV | VR as an induction medium, with EEG and HRV fusion. |
| 2019_Ehrlich_ClosedLoop_Music_BCI_Emotion_Mediation | The closest prior system to yours: closed-loop EEG to real-time synthesized affective music. |
| 2019_FernandezAguilar_Films_Emotion_MetaAnalysis | Effect sizes for film induction of positive and negative states, for power analysis and choosing stimuli. |

### Cornerstone papers not downloaded (paywalled; get them through UTCN library or the authors)
- Bradley & Lang (1994). Measuring emotion: the Self-Assessment Manikin and the semantic differential. J Behav Ther Exp Psychiatry 25(1). doi:10.1016/0005-7916(94)90063-9
- Lang, Bradley & Cuthbert (2008). IAPS: Affective ratings of pictures and instruction manual. Tech. Rep. A-8, Univ. of Florida (request from CSEA).
- Westermann et al. (1996). Relative effectiveness and validity of mood induction procedures: a meta-analysis. Eur J Soc Psychol 26(4):557-580.
- Kirschbaum, Pirke & Hellhammer (1993). The Trier Social Stress Test. Neuropsychobiology 28. doi:10.1159/000119004
- Pope, Bogart & Bartolome (1995). Biocybernetic system evaluates indices of operator engagement in automated task. Biol Psychol 40. doi:10.1016/0301-0511(95)05116-3. This is where the "biocybernetic loop" comes from.
- Gross (1998). The emerging field of emotion regulation: an integrative review. Rev Gen Psychol 2(3). doi:10.1037/1089-2680.2.3.271
- Juslin & Västfjäll (2008). Emotional responses to music: the need to consider underlying mechanisms. Behav Brain Sci 31(5). doi:10.1017/S0140525X08005293
- Fairclough (2009). Fundamentals of physiological computing. Interacting with Computers 21(1-2). doi:10.1016/j.intcom.2008.10.011
- Kreibig (2010). Autonomic nervous system activity in emotion: a review. Biol Psychol 84(3). doi:10.1016/j.biopsycho.2010.03.010
- Schaefer et al. (2010). Assessing the effectiveness of a large database of emotion-eliciting films. Cogn Emot 24(7). doi:10.1080/02699930903274322
- Siedlecka & Denson (2019). Experimental methods for inducing basic emotions: a qualitative review. Emotion Review 11(1). doi:10.1177/1754073917749016
- Joseph et al. (2020). The manipulation of affect: a meta-analysis of affect induction procedures. Psychol Bull 146(4). doi:10.1037/bul0000224

## 02_Recent_2021-2026
| File | Why it matters for your project |
|---|---|
| 2022_Somarathna_VR_for_Emotion_Elicitation_Review | Systematic review of VR as an elicitation method, with a taxonomy of designs and their validation. |
| 2022_Saganowski_Emognition_Dataset | Film induction with wearables (BVP, EDA) and self-report; a protocol template for multimodal recording. |
| 2022_Romeo_Slides_vs_Clips_Emotion_Induction_EEG | Static vs dynamic stimuli compared with EEG alpha; informs stimulus-modality choice. |
| 2022_FekriAzgomi_Faghih_ClosedLoop_Stress_Regulation_Control | State-space decoding of arousal from EDA feeding a feedback controller. Very close to your Kalman + controller design. |
| 2023_Chen_FACED_FinerGrained_Affective_EEG_Dataset | 123 subjects, 28 film clips, nine emotion categories; current best practice for large-scale film elicitation. |
| 2023_Agres_AffectMachine_Classical | Rule-based generator that outputs music at a target valence and arousal, i.e. a controllable music actuator. |
| 2024_Khazaei_Faghih_Bayesian_Arousal_Inference_Music | Bayesian filtering of hidden arousal while music is playing; the estimator side of a music loop. |
| 2025_Agres_AffectMachine_Pop_RealTime_Affective_Music | Real-time actuator, validated in a listening study, built for closed-loop use. |
| 2025_GutierrezMaestro_Affective_Priming_Score_Sequential_Elicitation | Detects carry-over (priming) between sequential stimuli in SEED. Directly relevant to your interleaved-quadrant protocol and the session confound. |
| 2025_Li_Designing_Light_for_Emotion_Color_Illuminance | EEG and physiology responses to light colour and illuminance; evidence for lighting as an actuator. |
| 2025_Sayal_Music_in_the_Loop_Neurofeedback_Review | Systematic review of music-based neurofeedback loops and their methodological gaps. |
| 2026_Humaidan_RL_Based_ClosedLoop_EEG_TMS | First working RL-controlled EEG-closed-loop stimulation, a reference for the RL controller option. |
| 2026_Kuang_Interaction_Effects_Emotion_in_VR | Interactivity amplifies and modulates VR-induced emotion (N=84). |
| 2026_Zhang_ClosedLoop_EEG_Neurofeedback_BCI_MentalHealth_Review | PRISMA review of closed-loop EEG interventions since 2021, covering 25 studies. |
| 2026_Zhang_MindMelody_ClosedLoop_EEG_Music_Intervention | Recent closed-loop EEG to generative-music system. |

### Recent paper worth getting (paywalled)
- Scangos et al. (2021). Closed-loop neuromodulation in an individual with treatment-resistant depression. Nat Med 27. doi:10.1038/s41591-021-01480-w
