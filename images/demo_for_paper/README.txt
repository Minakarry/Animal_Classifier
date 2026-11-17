论文截图用图（已从测试集筛选并复制）
01_success_* : 高置信度正确，适合成功案例
02_topk_*    : 正确且 Top-2 也有分量，适合展示 Top-3
03_fail_*    : 误分类，适合失败分析 / 图7

01_success_horse_100pct.jpeg	true=horse	pred=Horse	conf=1.000	top3=[('Horse', 1.0), ('Dog', 1.0179898187568726e-15), ('Elephant', 3.120555073596303e-17)]
01_success_squirrel_100pct.jpeg	true=squirrel	pred=Squirrel	conf=1.000	top3=[('Squirrel', 1.0), ('Spider', 2.929123957273738e-11), ('Butterfly', 3.9977712167158647e-14)]
01_success_horse_100pct_3.jpeg	true=horse	pred=Horse	conf=1.000	top3=[('Horse', 0.9999055862426758), ('Dog', 7.997862849151716e-05), ('Cow', 1.3419413335213903e-05)]
01_success_sheep_100pct.jpg	true=sheep	pred=Sheep	conf=0.998	top3=[('Sheep', 0.9984084963798523), ('Elephant', 0.0012190876295790076), ('Squirrel', 0.0002903984277509153)]
01_success_spider_100pct.jpeg	true=spider	pred=Spider	conf=1.000	top3=[('Spider', 1.0), ('Dog', 5.278986847656597e-08), ('Squirrel', 1.4819112603703388e-09)]
01_success_butterfly_100pct.jpeg	true=butterfly	pred=Butterfly	conf=1.000	top3=[('Butterfly', 0.9999985694885254), ('Chicken', 6.964203294046456e-07), ('Spider', 4.938162305734295e-07)]
01_success_elephant_100pct.jpeg	true=elephant	pred=Elephant	conf=1.000	top3=[('Elephant', 1.0), ('Cat', 1.4112447876968304e-09), ('Dog', 8.951667740397795e-10)]
01_success_butterfly_100pct_8.jpeg	true=butterfly	pred=Butterfly	conf=1.000	top3=[('Butterfly', 1.0), ('Spider', 5.303335353333549e-13), ('Horse', 1.5536754012241555e-13)]
01_success_sheep_100pct.jpeg	true=sheep	pred=Sheep	conf=1.000	top3=[('Sheep', 1.0), ('Dog', 3.33430991150685e-11), ('Horse', 2.2067549176885137e-11)]
01_success_horse_100pct_10.jpeg	true=horse	pred=Horse	conf=1.000	top3=[('Horse', 0.9999881982803345), ('Spider', 1.0419121281302068e-05), ('Cow', 1.3587078910859418e-06)]
01_success_butterfly_100pct.jpg	true=butterfly	pred=Butterfly	conf=1.000	top3=[('Butterfly', 1.0), ('Spider', 3.515144379457984e-14), ('Horse', 5.776773170603801e-15)]
01_success_cow_100pct.jpeg	true=cow	pred=Cow	conf=1.000	top3=[('Cow', 0.9999998807907104), ('Horse', 1.2506833968473074e-07), ('Sheep', 1.9907395731877386e-09)]
02_topk_butterfly_vs_Spider_89pct.jpeg	true=butterfly	pred=Butterfly	conf=0.889	top3=[('Butterfly', 0.8894949555397034), ('Spider', 0.11002950370311737), ('Squirrel', 0.00047247251495718956)]
02_topk_sheep_vs_Chicken_67pct.jpeg	true=sheep	pred=Sheep	conf=0.670	top3=[('Sheep', 0.6700170636177063), ('Chicken', 0.257621705532074), ('Horse', 0.046530336141586304)]
03_fail_true-dog_pred-Elephant_90pct.jpeg	true=dog	pred=Elephant	conf=0.896	top3=[('Elephant', 0.8960760831832886), ('Dog', 0.10392389446496964), ('Horse', 3.0366269498927068e-09)]
03_fail_true-cow_pred-Cat_100pct.jpeg	true=cow	pred=Cat	conf=1.000	top3=[('Cat', 1.0), ('Dog', 4.3535567512131834e-11), ('Squirrel', 3.50211464103696e-14)]
