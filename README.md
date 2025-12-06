**Here is our implementation of approach built for the paper named:**
### Analyzing and Predicting Diabetes with Deep Learning and Visual Insights.

### **Dataset Overview:**
Our dataset comprises 89 features and 561 rows to categorize U.S. states into three diabetes prevalence groups: low (6.0%–8.9%), medium (9.0%–10.9%), and high (11.0%–13.9%). 

The dataset used in this study contains 89 features, representing annual percentages for each U.S. state from 2011 to 2021. The first category, Diabetes, captures the percentage of adults aged 18 and older who have been diagnosed with diabetes. Although this variable does not contribute to the feature count, it serves as the primary target indicator in our analysis.

The second and largest category, Chronic Disease Indicators, consists of 71 features. These represent the percentage of adults diagnosed with at least one chronic disease (excluding diabetes) or exhibiting related health indicators. Examples include asthma, arthritis, kidney disease, high cholesterol, smoking, and the receipt of recommended examinations such as foot checks, dilated eye exams, or glycosylated hemoglobin measurements.

The dataset also incorporates demographic information. The Races category contains five features describing the percentage distribution across major racial groups, including Hispanic, Non-Hispanic White, Non-Hispanic Asian or Pacific Islander, Non-Hispanic American Indian or Alaska Native, and Non-Hispanic Black populations. Similarly, the Age Groups category includes four features that represent the population distribution across age brackets: 0–19, 20–39, 40–59, and 60 and older.

Additional population characteristics are captured through the Gender, House, and Economy categories. The Gender category comprises three features detailing the percentages of males, females, and the total population. The House category includes three features representing the proportions of total, vacant, and occupied housing units. Finally, the Economy category also contains three features, covering the percentage of the employed population, per capita income, and the poverty rate.

Collectively, these categories contribute to a total of 89 features, encompassing a comprehensive set of demographic, socioeconomic, and health-related indicators for each state across the 11-year study period.

### Metadata Explanation:
1. Year: Year from 2011 to 2021: Year
2. StateName: Name of US State: 
3. StateAbbr: Abbreviate of the name of US State: 
4. AG0_4: Age group data with population from 0-19 years old (%).
5. AG5_8: Age group data with population from 20-39 years old (%).
6. AG9_12: Age group data with population from 40-59 years old (%).
7. AG13_18: Age group data with population from over 60 years old (%).
8. N_WHT: Non-Hispanic White (%).
9. N_BLK: Non-Hispanic Black (%).
10. N_AIAN: Non-Hispanic American Indian/Alaska Native (%).
11. N_ASNPI: Non-Hispanic Asian or Pacific Islander (%).
12. N_ALL: Hispanic (All Races) (%).
13. MALE: the number of males in the state (%).
14. FEMALE: the number of females in the state (%).
15. POP: The population of state in the year (%).
16. VA_HOU: Vacant houses of the state (%).
17. OC_HOU: Occupied houses of the state (%).
18. TO_HOU: total number of state's houses (%).
19. EM_PER: Employed population of the state (%).
20. PCP_INC: Per capita income of the state's population (%).
21. PO_RATE: Poverty rate of the state's population (%).
22. ALC2_2: Binge drinking prevalence among adults aged >= 18 years (%).
23. ALC5_1: Heavy drinking among adults aged >= 18 years (%).
24. ART1_1: Arthritis among adults aged >= 18 years (%).
25. ART1_2: Arthritis among adults aged >= 18 years who are obese (%).
26. ART1_3: Arthritis among adults aged >= 18 years who have diabetes (%).
27. ART1_4: Arthritis among adults aged >= 18 years who have heart disease (%).
28. ART2_1: Activity limitation due to arthritis among adults aged >= 18 years who have doctor-diagnosed arthritis (%).
29. ART2_2: Severe joint pain due to arthritis among adults aged >= 18 years who have doctor-diagnosed arthritis (%).
30. ART2_3: Work limitation due to arthritis among adults aged 18-64 years who have doctor-diagnosed arthritis (%).
31. ART3_0: Physical inactivity among adults aged >= 18 years with arthritis (%).
32. ART4_0: Fair or poor health among adults aged >= 18 years with arthritis (%).
33. ART5_0: Adults aged >= 18 years with arthritis who have taken a class to learn how to manage arthritis symptoms (%).
34. AST1_1: Current asthma prevalence among adults aged >= 18 years (%).
35. AST5_1: Influenza vaccination among noninstitutionalized adults aged 18-64 years with asthma (%).
36. AST5_2: Influenza vaccination among noninstitutionalized adults aged >= 65 years with asthma (%).
37. AST6_1: Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years with asthma (%).
38. AST6_2: Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years with asthma (%).
39. CAN1_0: Mammography use among women aged 50-74 years (%).
40. CAN2_1: Papanicolaou smear use among adult women aged 21-65 years (%).
41. CAN3_0: Fecal occult blood test, sigmoidoscopy, or colonoscopy among adults aged 50-75 years (%).
42. CKD3_0: Prevalence of chronic kidney disease among adults aged >= 18 years (%).
43. COPD2_0: Prevalence of chronic obstructive pulmonary disease among adults >= 18 (%).
44. COPD2_0_1: Prevalence of chronic obstructive pulmonary disease among adults >= 45 years (%).
45. COPD3_0: Prevalence of current smoking among adults >= 18 with diagnosed chronic obstructive pulmonary disease (%).
46. COPD3_0_1: Prevalence of current smoking among adults >= 45 years with diagnosed chronic obstructive pulmonary disease (%).
47. COPD4_0: Prevalence of activity limitation among adults >= 18 with diagnosed chronic obstructive pulmonary disease (%).
48. COPD4_0_1: Prevalence of activity limitation among adults >= 45 years with diagnosed chronic obstructive pulmonary disease (%).
49. COPD7_0: Influenza vaccination among noninstitutionalized adults aged >= 45 years with chronic obstructive pulmonary disease (%).
50. COPD8_0: Pneumococcal vaccination among noninstitutionalized adults aged >= 45 years with chronic obstructive pulmonary disease (%).
51. CVD10_1: Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years with a history of coronary heart disease (%).
52. CVD10_2: Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years with a history of coronary heart disease (%).
53. CVD4_0: Cholesterol screening among adults aged >= 18 years (%).
54. CVD5_0: High cholesterol prevalence among adults aged >= 18 years (%).
55. CVD6_1: Awareness of high blood pressure among adults aged >= 18 years (%).
56. CVD7_0: Taking medicine for high blood pressure control among adults aged >= 18 years with high blood pressure (%).
57. CVD9_1: Influenza vaccination among noninstitutionalized adults aged 18-64 years with a history of coronary heart disease or stroke (%).
58. CVD9_2: Influenza vaccination among noninstitutionalized adults aged >= 65 years with a history of coronary heart disease or stroke (%).
59. DIA10_0: Adults with diagnosed diabetes aged >= 18 years who have taken a diabetes self-management course (%).
60. DIA11_1: Prevalence of high cholesterol among adults aged >= 18 years with diagnosed diabetes (%).
61. DIA11_2: Prevalence of high blood pressure among adults aged >= 18 years with diagnosed diabetes (%).
62. DIA11_3: Prevalence of depressive disorders among adults aged >= 18 years with diagnosed diabetes (%).
63. DIA12_1: Influenza vaccination among noninstitutionalized adults aged 18-64 years with diagnosed diabetes (%).
64. DIA12_2: Influenza vaccination among noninstitutionalized adults aged >= 65 years with diagnosed diabetes (%).
65. DIA13_1: Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years with diagnosed diabetes (%).
66. DIA13_2: Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years with diagnosed diabetes (%).
67. DIA2_1: Prevalence of diagnosed diabetes among adults aged >= 18 years (%).
68. DIA5_0: Foot examination among adults aged >= 18 years with diagnosed diabetes (%).
69. DIA6_0: Glycosylated hemoglobin measurement among adults aged >= 18 years with diagnosed diabetes (%).
70. DIA7_0: Dilated eye examination among adults aged >= 18 years with diagnosed diabetes (%).
71. DIA8_0: Visits to dentist or dental clinic among adults aged >= 18 years with diagnosed diabetes (%).
72. IMM1_0: Influenza vaccination among noninstitutionalized adults aged >= 18 years (%).
73. NPAW1_1: Obesity among adults aged >= 18 years (%).
74. NPAW10_0: No leisure-time physical activity among adults aged >= 18 years (%).
75. NPAW11_1: Meeting aerobic physical activity guidelines for substantial health benefits among adults aged >= 18 years (%).
76. NPAW11_2: Meeting aerobic physical activity guidelines for substantial health benefits and for muscle-strengthening activity among adults aged >= 18 years (%).
77. NPAW11_3: Meeting aerobic physical activity guidelines for additional and more extensive health benefits among adults aged >= 18 years (%).
78. NPAW2_1: Overweight or obesity among adults aged >= 18 years (%).
79. NPAW3_1: Healthy weight among adults aged >= 18 years (%).
80. OLD3_1: Proportion of older adults aged >= 65 years who are up to date on a core set of clinical preventive services (%).
81. OLD3_2: Proportion of older adults aged 50-64 years who are up to date on a core set of clinical preventive services (%).
82. ORH1_1: Visits to dentist or dental clinic among adults aged >= 18 years (%).
83. ORH4_1: All teeth lost among adults aged >= 65 years (%).
84. ORH4_2: Six or more teeth lost among adults aged >= 65 years (%).
85. ORH4_3: No tooth loss among adults aged 18-64 years (%).
86. OVC1_1: Current lack of health insurance among adults aged 18-64 years (%).
87. OVC6_1: Fair or poor self-rated health status among adults aged >= 18 years (%).
88. OVC8_0: Prevalence of sufficient sleep among adults aged >= 18 years (%).
89. TOB1_2: Current smoking among adults aged >= 18 years (%).
90. TOB11_1: Pneumococcal vaccination among noninstitutionalized adults aged 18-64 years who smoke (%).
91. TOB11_2: Pneumococcal vaccination among noninstitutionalized adults aged >= 65 years who smoke (%).
92. TOB2_2: Current smokeless tobacco use among adults aged >= 18 years (%).
93. TOB3_0: Quit attempts in the past year among current smokers (%).
