from cmath import exp, sqrt, nan
import pandas as pd
import numpy as np

class date_n:
    def __init__(self, day, month, year):
        self.day = day
        self.month = months.index(month) + 1
        self.year = year

month_Days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def count_Leap_Years(day):

    years = day.year

    if (day.month <= 2):
        years -= 1

    return int(years / 4) - int(years / 100) + int(years / 400)

def get_difference(date_1, date_2):

    n_1 = date_1.year * 365 + date_1.day

    for K in range(0, date_1.month - 1):
        n_1 += month_Days[K]

    n_1 += count_Leap_Years(date_1)

    n_2 = date_2.year * 365 + date_2.day
    for K in range(0, date_2.month - 1):
        n_2 += month_Days[K]
    n_2 += count_Leap_Years(date_2)

    return (n_2 - n_1)

def price_xyz(A_t_d, B, t_e, r, sigma, t_d, numsteps):
    date_1 = date_n(int(t_d[0] + t_d[1]), t_d[2] + t_d[3] + t_d[4], int(t_d[5] + t_d[6] + t_d[7] + t_d[8]))
    date_2 = date_n(int(t_e[0] + t_e[1]), t_e[2] + t_e[3] + t_e[4], int(t_e[5] + t_e[6] + t_e[7] + t_e[8]))
    no_days = get_difference(date_1, date_2)
    delta_t = float(no_days) / float(numsteps)
    if delta_t >= 2*pow(sigma,2)/pow(r,2):
        return -1
    p_u_num = exp(r * delta_t/2) - exp(-sigma * sqrt(delta_t/2))
    p_d_num = exp(r * delta_t/2) - exp(sigma * sqrt(delta_t/2))
    p_den = p_d_num - p_u_num
    p_u = pow(p_u_num, 2) / pow(p_den, 2).real
    p_d = pow(p_d_num, 2) / pow(p_den, 2).real
    j_u = exp(sigma*sqrt(delta_t)).real
    j_d = exp(-sigma*sqrt(delta_t)).real
    
    Costs = [A_t_d]
    for i in range(0, numsteps):
        Costs.append(Costs[-1]*j_u)
        Costs.insert(0, Costs[0]*j_d)

    for i in range(0, len(Costs)):
        Costs[i] = max(Costs[i] - B, 0.0)
    for i in range(0, numsteps):
        newlist = []
        for j in range(1, len(Costs) - 1):
            newlist.append((p_u*Costs[j+1] + p_d*Costs[j-1] + Costs[j]*(1 - p_u - p_d))*(exp(-r*delta_t).real))
        Costs = newlist

    return Costs[0].real

def volrisk_xyz(A_t_d, B, t_e, r, sigma, t_d, numsteps):
    date_1 = date_n(int(t_d[0] + t_d[1]), t_d[2] + t_d[3] + t_d[4], int(t_d[5] + t_d[6] + t_d[7] + t_d[8]))
    date_2 = date_n(int(t_e[0] + t_e[1]), t_e[2] + t_e[3] + t_e[4], int(t_e[5] + t_e[6] + t_e[7] + t_e[8]))
    no_days = get_difference(date_1, date_2)
    delta_t = float(no_days) / float(numsteps)
    if delta_t >= 2*pow(sigma,2)/pow(r,2):
        return -1
    p_u_num = exp(r * delta_t/2) - exp(-sigma * sqrt(delta_t/2))
    p_d_num = exp(r * delta_t/2) - exp(sigma * sqrt(delta_t/2))
    p_den = p_d_num - p_u_num
    p_u = pow(p_u_num, 2) / pow(p_den, 2).real
    p_d = pow(p_d_num, 2) / pow(p_den, 2).real

    j_u = exp(sigma*sqrt(delta_t)).real
    j_d = exp(-sigma*sqrt(delta_t)).real

    check = (price_xyz(A_t_d, B, t_e, r, sigma + 0.00001, t_d, numsteps) - price_xyz(A_t_d, B, t_e, r, sigma, t_d, numsteps))/0.00001

    d_p_u = -(sqrt(2*delta_t)*(exp(sqrt(delta_t/2)*sigma +r*delta_t/2) - 1)*(exp(3*sqrt(delta_t/2)*sigma +r*delta_t/2) + exp(sqrt(delta_t/2)*sigma +r*delta_t/2) - 2*exp(sqrt(2*delta_t)*sigma)))/((exp(sqrt(2*delta_t)*sigma) - 1) ** 3)
    d_p_d = -(sqrt(2*delta_t)*(exp(sqrt(delta_t/2)*sigma -r*delta_t/2) - 1)*((exp(r*delta_t/2)*(-exp(sqrt(2*delta_t)*sigma) - 1) + 2*exp(sqrt(delta_t/2)*sigma))*exp(sqrt(2*delta_t)*sigma)))/((exp(sqrt(2*delta_t)*sigma) - 1) ** 3)
    Costs = [A_t_d]
    for i in range(0, numsteps):
        Costs.append(Costs[-1]*j_u)
        Costs.insert(0, Costs[0]*j_d)

    differentiations = []
    for i in range(len(Costs)):
        if Costs[i] >= B:
            differentiations.append((A_t_d)*((i - numsteps)*sqrt(delta_t)*(exp((i - numsteps)*sigma*sqrt(delta_t)))).real)
        else:
            differentiations.append(0)

    for i in range(0, len(Costs)):
        Costs[i] = max(Costs[i] - B, 0.0)

    d_p_d = d_p_d.real
    d_p_u = d_p_u.real

    for i in range(0, numsteps):
        newlist = []
        newlist2 = []
        for j in range(1, len(Costs) - 1):
            newlist.append(((p_u*Costs[j+1] + p_d*Costs[j-1] + Costs[j]*(1 - p_u - p_d))*(exp(-r*delta_t).real)).real)
            newlist2.append((((d_p_u*Costs[j+1] + p_u*differentiations[j+1] - d_p_u*Costs[j] - d_p_d*Costs[j] + (1 - p_d - p_u)*differentiations[j] + d_p_d*Costs[j-1] + p_d*differentiations[j-1]))*(exp(-r*delta_t).real)).real)
        Costs = newlist
        differentiations = newlist2

    if abs(check - differentiations[0])/abs(check) < 0.1:
        return differentiations[0]
    else: 
        return check
    
ans = []
ans.append(price_xyz(317, 297, "23Sep2023", 0.035, 0.025, "28Jun2022", 1000))
ans.append(price_xyz(190, 230, "01Aug2023", 0.015, 0.017, "12Jan2023", 2000))
ans.append(price_xyz(100, 100, "12Sep2022", -0.015, 0.020, "26Jun2022", 1500))
ans.append(volrisk_xyz(317, 297, "23Sep2023", 0.035, 0.025, "28Jun2022", 1000))
ans.append(volrisk_xyz(190, 230, "01Aug2023", 0.015, 0.017, "12Jan2023", 2000))
ans.append(volrisk_xyz(100, 100, "12Sep2022", -0.015, 0.020, "26Jun2022", 1500))

df = pd.DataFrame(ans)
df.to_csv('ans.csv')