\ENCODING=ISO-8859-1
\Problem name: IloCplex

Maximize
 obj1: 0.5 qL_cond_74 + 0.5 qL_cond_73 - 1
Subject To
 c1:  - 0.1 qL_cond_74 - 0.1 qL_cond_73 + qL_0 - 0.1 qL_cond_72
      - 0.1 qL_cond_75 - 0.1 qL_cond_76  = 1
 c2:  qF_1 - 0.1 qF_cond_62 - 0.1 qF_cond_63 - 0.1 qF_cond_64 - 0.1 qF_cond_65
      - 0.1 qF_cond_66 - 0.1 qF_cond_67 - 0.1 qF_cond_68 - 0.1 qF_cond_69
      - 0.1 qF_cond_70 - 0.1 qF_cond_71  = 1
 c3:  wF_2 + wF_3 + wF_4 + wF_5 + wF_6 + wF_7  = 1
 c4:  wF_8 + wF_9 + wF_10 + wF_11 + wF_12 + wF_13  = 1
 c5:  wF_14 + wF_15 + wF_16 + wF_17 + wF_18 + wF_19  = 1
 c6:  wF_20 + wF_21 + wF_22 + wF_23 + wF_24 + wF_25  = 1
 c7:  wF_26 + wF_27 + wF_28 + wF_29 + wF_30 + wF_31  = 1
 c8:  wF_32 + wF_33 + wF_34 + wF_35 + wF_36 + wF_37  = 1
 c9:  wF_38 + wF_39 + wF_40 + wF_41 + wF_42 + wF_43  = 1
 c10: wF_44 + wF_45 + wF_46 + wF_47 + wF_48 + wF_49  = 1
 c11: wF_50 + wF_51 + wF_52 + wF_53 + wF_54 + wF_55  = 1
 c12: wF_56 + wF_57 + wF_58 + wF_59 + wF_60 + wF_61  = 1
 c13: sigma_L_0 + sigma_L_1  = 1
 c14: sigma_L_2 + sigma_L_3  = 1
Bounds
-1000 <= qL_cond_74 <= 1000
-1000 <= qL_cond_73 <= 1000
-1000 <= qL_0 <= 1000
-1000 <= qL_cond_72 <= 1000
-1000 <= qL_cond_75 <= 1000
-1000 <= qL_cond_76 <= 1000
-1000 <= qF_1 <= 1000
-1000 <= qF_cond_62 <= 1000
-1000 <= qF_cond_63 <= 1000
-1000 <= qF_cond_64 <= 1000
-1000 <= qF_cond_65 <= 1000
-1000 <= qF_cond_66 <= 1000
-1000 <= qF_cond_67 <= 1000
-1000 <= qF_cond_68 <= 1000
-1000 <= qF_cond_69 <= 1000
-1000 <= qF_cond_70 <= 1000
-1000 <= qF_cond_71 <= 1000
 0 <= wF_2 <= 1
 0 <= wF_3 <= 1
 0 <= wF_4 <= 1
 0 <= wF_5 <= 1
 0 <= wF_6 <= 1
 0 <= wF_7 <= 1
 0 <= wF_8 <= 1
 0 <= wF_9 <= 1
 0 <= wF_10 <= 1
 0 <= wF_11 <= 1
 0 <= wF_12 <= 1
 0 <= wF_13 <= 1
 0 <= wF_14 <= 1
 0 <= wF_15 <= 1
 0 <= wF_16 <= 1
 0 <= wF_17 <= 1
 0 <= wF_18 <= 1
 0 <= wF_19 <= 1
 0 <= wF_20 <= 1
 0 <= wF_21 <= 1
 0 <= wF_22 <= 1
 0 <= wF_23 <= 1
 0 <= wF_24 <= 1
 0 <= wF_25 <= 1
 0 <= wF_26 <= 1
 0 <= wF_27 <= 1
 0 <= wF_28 <= 1
 0 <= wF_29 <= 1
 0 <= wF_30 <= 1
 0 <= wF_31 <= 1
 0 <= wF_32 <= 1
 0 <= wF_33 <= 1
 0 <= wF_34 <= 1
 0 <= wF_35 <= 1
 0 <= wF_36 <= 1
 0 <= wF_37 <= 1
 0 <= wF_38 <= 1
 0 <= wF_39 <= 1
 0 <= wF_40 <= 1
 0 <= wF_41 <= 1
 0 <= wF_42 <= 1
 0 <= wF_43 <= 1
 0 <= wF_44 <= 1
 0 <= wF_45 <= 1
 0 <= wF_46 <= 1
 0 <= wF_47 <= 1
 0 <= wF_48 <= 1
 0 <= wF_49 <= 1
 0 <= wF_50 <= 1
 0 <= wF_51 <= 1
 0 <= wF_52 <= 1
 0 <= wF_53 <= 1
 0 <= wF_54 <= 1
 0 <= wF_55 <= 1
 0 <= wF_56 <= 1
 0 <= wF_57 <= 1
 0 <= wF_58 <= 1
 0 <= wF_59 <= 1
 0 <= wF_60 <= 1
 0 <= wF_61 <= 1
 0 <= sigma_L_0 <= 1
 0 <= sigma_L_1 <= 1
 0 <= sigma_L_2 <= 1
 0 <= sigma_L_3 <= 1
Binaries
 wF_2  wF_3  wF_4  wF_5  wF_6  wF_7  wF_8  wF_9  wF_10  wF_11  wF_12  wF_13 
 wF_14  wF_15  wF_16  wF_17  wF_18  wF_19  wF_20  wF_21  wF_22  wF_23  wF_24 
 wF_25  wF_26  wF_27  wF_28  wF_29  wF_30  wF_31  wF_32  wF_33  wF_34  wF_35 
 wF_36  wF_37  wF_38  wF_39  wF_40  wF_41  wF_42  wF_43  wF_44  wF_45  wF_46 
 wF_47  wF_48  wF_49  wF_50  wF_51  wF_52  wF_53  wF_54  wF_55  wF_56  wF_57 
 wF_58  wF_59  wF_60  wF_61 
End
