\ENCODING=ISO-8859-1
\Problem name: IloCplex

Maximize
 obj1:
Subject To
 c1: qF_2 - qF_3 >= 0
 c2: qL_0  = 0
 c3: qF_2  = 0
 c4: qF_3  = 0
Bounds
-1000 <= qF_2 <= 1000
-1000 <= qF_3 <= 1000
-1000 <= qL_0 <= 1000
End
