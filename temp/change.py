test = "-0.001735   0.855388   0.315499   2.008551   7.606260  -0.798294  11.216058  -3.286777  -1.592436  13.521250  -1.153514  -4.213484 -17.754157  -3.216621   9.232892  -7.948705   0.211932  -1.528529   2.220789  -0.981058  -1.133630   2.071938  -6.311876   2.083844   2.020309  -0.533885 -19.342332  -5.129554 -37.575293 -50.190804   0.198025 -24.741038   4.442069   0.442380   2.547494   4.858004   1.951773  -5.809334  21.100535  23.710456  30.003467  53.240376   0.414981  10.414544   1.952633   3.576914  -9.482057   6.918939   1.457480  -0.035296   0.111891 -27.722826  -1.655032   2.430426  -2.964232  -5.507982   1.444119   2.239212  -3.180259  -0.892285  -0.008100  -0.007000   0.024400"
print(test.strip)
test = test.split(" ")
test = [i for i in test if i]
print(test)
print(len(test))
# 25
# 63 - 3 = 60
# 60 / 3 = 20
# 20 = 25 - 5*end_site


# 生成87行数据，第一列和第二列互补，其他列为0，输出5位小数
rows = 4000

stands = [[0, 159]]
transitions = [[160, 192]]
runs = [[193, 3999]]
data = []

for i in range(rows):
    for stand in stands:
        if i >= stand[0] and i <= stand[1]:
            x = (i - stand[0]) / (stand[1] - stand[0])
            stand_col = '1.00000'  # 第一列从1到0，格式化为5位小数
            run_col = '0.00000'        # 第二列从0到1，格式化为5位小数
            other_cols = ['0.00000'] * 6                        # 其他6列为0，格式化为5位小数
            break
    for run in runs:
        if i >= run[0] and i <= run[1]:
            x = (i - run[0]) / (run[1] - run[0])
            stand_col = '0.00000'  # 第一列从1到0，格式化为5位小数
            run_col = '1.00000'        # 第二列从0到1，格式化为5位小数
            other_cols = ['0.00000'] * 6                        # 其他6列为0，格式化为5位小数
            break    
    for transition in transitions:
        if i >= transition[0] and i <= transition[1]:
            x = (i - transition[0]) / (transition[1] - transition[0])
            stand_col = format(1 - x, '.5f')  # 第一列从1到0，格式化为5位小数
            run_col = format(x, '.5f')        # 第二列从0到1，格式化为5位小数
            other_cols = ['0.00000'] * 6                        # 其他6列为0，格式化为5位小数
            break

    data.append([stand_col, run_col] + other_cols)

# 输出结果到txt文件
with open('output.txt', 'w') as f:
    for row in data:
        f.write(" ".join(row) + "\n")