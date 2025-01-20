from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize
import random
import time

def create_adjusted_natural_2d_array(start, end, rows, cols, zero_ratio=0.8):

    if start < 1 or end < 1:
        raise ValueError('開始値と終了値は自然数である必要があります。')

    if start > end:
        raise ValueError('開始値と終了値より小さくなければなりません')
    
    if rows <= 0 or cols <= 0:
        raise ValueError('行と列は正の整数である必要があります。')
    
    if not (0 <= zero_ratio <= 1):
        raise ValueError('zero_ratioは0から1までの値である必要があります')
    
    array = []

    for _ in range(rows):
        #行ごとの要素を生成
        row = [random.randint(start, end) for _ in range(cols)]

        #ゼロにする要素
        num_zeros = int(cols*zero_ratio)

        #ゼロにする位置をランダム選択
        indices_to_zero = random.sample(range(cols), num_zeros)

        for index in indices_to_zero:
            row[index]=0
        array.append(row)
    
    return array


numa, numb, numc = 50, 250, 100000
P_Aa, P_Ab, P_Ac = 16395, 16395, 13482
P_Ba, P_Bb, P_Bc = 8000, 0, 24000

problem = LpProblem("Integer_Programming_Example", LpMinimize)

T_Aa=LpVariable('T_Aa',lowBound=0)
T_Ab=LpVariable('T_Ab',lowBound=0)
T_Ac=LpVariable('T_Ac',lowBound=0)
T_Ba=LpVariable('T_Ba',lowBound=0)
T_Bb=LpVariable('T_Bb',lowBound=0)
T_Bc=LpVariable('T_Bc',lowBound=0)

problem += T_Aa+T_Ab+T_Ac+T_Ba+T_Bb+T_Bc, "Objective"

problem += P_Aa*T_Aa + P_Ba*T_Ba >= numa, f"Constraint_1"
problem += P_Ab*T_Ab + P_Bb*T_Bb >= numb, f"Constraint_2"
problem += P_Ac*T_Ac + P_Bc*T_Bc >= numc, f"Constraint_3"

problem.solve()

print(f"Optimal value for T_Aa:", value(T_Aa))
print(f"Optimal value for T_Ab:", value(T_Ab))
print(f"Optimal value for T_Ac:", value(T_Ac))
print(f"Optimal value for T_Ba:", value(T_Ba))
print(f"Optimal value for T_Bb:", value(T_Bb))
print(f"Optimal value for T_Bc:", value(T_Bc))

c=0

"""#定数
PRODUCT_NUM = 6
MACHINE_NUM = 6

#入力（H50単位の生産量、製品数の1次元配列）
min_mount = 5
max_mount = 1000
product_mount = [random.randint(min_mount,max_mount) for _ in range(PRODUCT_NUM)]

#定数（設備生産能力、製品数x設備数の行列）
min_spec = 100
max_spec = 20000
zero_ratio = 0.6
machine_product_spec = create_adjusted_natural_2d_array(min_spec, max_spec, PRODUCT_NUM, MACHINE_NUM, zero_ratio)

problem = LpProblem("Integer_Programming_Example", LpMinimize)

tmps = [f"product_{i+1}_machine_{j+1}" for i in range(len(machine_product_spec)) for j in range(len(machine_product_spec[i]))]

objective_function = None
for index, tmp in enumerate(tmps):
    tmps[index] = LpVariable(tmp, lowBound=0)
    objective_function += tmps[index]


problem += objective_function, "Objective"

constraints = [None] * PRODUCT_NUM
for i_index, product_constranint in enumerate(product_mount):
    for j_index in range(len(machine_product_spec[i_index])):
        constraints[i_index] += tmps[MACHINE_NUM*i_index+j_index] * machine_product_spec[i_index][j_index]

    problem += constraints[i_index] >= product_mount[i_index], f"Constraint_{i_index}"

problem.solve()

for index, tmp in enumerate(tmps):
    print(f"Optimal value for {tmps[index]}:", value(tmps[index]))

print("Minimum objective function value:", value(problem.objective))"""

check=0