from cmath import nan
import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle 

def predictInnings(inningFile,predictionFile, modelFile=''):
    
    df_test = pd.read_csv(inningFile)
    if (modelFile!=''):
        model = pd.read_pickle(modelFile)
    Total_runs = []
    Total_wickets = []
    Total_wides = []
    Total_noball = []
    for i in range(len(df_test)):
        if(df_test['over'][i] != 1 or df_test['ball'][i] != 1):
            Total_runs.append(Total_runs[i-1]+ df_test['total_runs'][i])
            if(df_test['dismissal_kind'][i] == "caught" or df_test['dismissal_kind'][i] == "bowled" or df_test['dismissal_kind'][i] == "lbw" or df_test['dismissal_kind'][i] == "stumped" or df_test['dismissal_kind'][i] == "run out"):
                Total_wickets.append(Total_wickets[i-1]+1)
            else:
                Total_wickets.append(Total_wickets[i-1])
            Total_wides.append(Total_wides[i-1]+ df_test['wide_runs'][i])
            Total_noball.append(Total_noball[i-1] + df_test['noball_runs'][i])
        else:
            if(df_test['dismissal_kind'][i] == "caught" or df_test['dismissal_kind'][i] == "bowled" or df_test['dismissal_kind'][i] == "lbw" or df_test['dismissal_kind'][i] == "stumped" or df_test['dismissal_kind'][i] == "run out"):
                Total_wickets.append(1)
            else:
                Total_wickets.append(0)
            Total_runs.append(df_test['total_runs'][i])
            Total_wides.append(df_test['wide_runs'][i])
            Total_noball.append(df_test['noball_runs'][i])
    df_test['Total_runs'] = Total_runs
    df_test['Total_wickets'] = Total_wickets
    df_test['Total_wides'] = Total_wides
    df_test['Total_noball'] = Total_noball

    rem_cols = ['batsman', 'non_striker', 'bowler', 'wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs', 'batsman_runs', 'extra_runs', 'total_runs', 'player_dismissed', 'dismissal_kind', 'fielder']    
    df_test.drop(labels=rem_cols , axis=1 , inplace = True)

    # Total_runs = []
    # Total_wides = []
    # Total_noball = []
    # Total_wickets = []
    # Match_total = []
    # balls = 0
    # for i in range(len(df)):
    #     if(df['over'][i] != 1 or df['ball'][i] != 1):
    #         balls += 1
    #         Total_runs.append(Total_runs[i-1]+ df['total_runs'][i])        
    #         if(df['dismissal_kind'][i] == "caught" or df['dismissal_kind'][i] == "bowled" or df['dismissal_kind'][i] == "lbw" or df['dismissal_kind'][i] == "stumped" or df['dismissal_kind'][i] == "run out"):
    #             Total_wickets.append(Total_wickets[i-1]+1)
    #         else:
    #             Total_wickets.append(Total_wickets[i-1])
    #         Total_wides.append(Total_wides[i-1]+ df['wide_runs'][i])
    #         Total_noball.append(Total_noball[i-1] + df['noball_runs'][i])
    #     else:
    #         total = []
    #         for i in range(balls):
    #             total.append(Total_runs[-1])
    #         Match_total.extend(total)
    #         balls = 1
    #         if(df['dismissal_kind'][i] == "caught" or df['dismissal_kind'][i] == "bowled" or df['dismissal_kind'][i] == "lbw" or df['dismissal_kind'][i] == "stumped" or df['dismissal_kind'][i] == "run out"):
    #             Total_wickets.append(1)
    #         else:
    #             Total_wickets.append(0)
    #         Total_runs.append(df['total_runs'][i])
    #         Total_wides.append(df['wide_runs'][i])
    #         Total_noball.append(df['noball_runs'][i])
    # total = []
    # for i in range(balls):
    #     total.append(Total_runs[-1])
    # Match_total.extend(total)
    # df['Total_runs'] = Total_runs
    # df['Total_wickets'] = Total_wickets
    # df['Match_total'] = Match_total
    # df['Total_wides'] = Total_wides
    # df['Total_noball'] = Total_noball

    # df.drop(labels=rem_cols , axis=1 , inplace = True)
    # df.to_pickle('model.pkl')
    # model = df
    model = model[(model['batting_team'].isin(df_test['batting_team'].unique())) & (model['bowling_team'].isin(df_test['bowling_team'].unique()))]
    model = model[model['over'] <= 10]
    cat_df = pd.get_dummies(data = model, columns = ['batting_team' , 'bowling_team'])
    cat_df_test = pd.get_dummies(data = df_test, columns = ['batting_team' , 'bowling_team'])
    x_train = cat_df.drop(labels = ['Match_total', 'match_id'], axis = 1)
    y_train = cat_df['Match_total']
    x_test = cat_df_test.drop(labels = 'match_id', axis = 1)
    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    prediction = reg.predict(x_test)
    predict_array = []
    for i in range(1, len(prediction)):
        if x_test['over'][i] == 1 and x_test['ball'][i] == 1:
            predict_array.append([cat_df_test['match_id'][i-1], int(prediction[i-1])])

    predict_array.append([cat_df_test['match_id'][len(prediction)-1], int(prediction[-1])])
    predict_df = pd.DataFrame(predict_array, columns = ['match_id', 'total_runs'])
    predict_df.to_csv(predictionFile, index = False)



predictInnings(inningFile='IPL_test.csv', predictionFile='test_prediction.csv', modelFile= 'model.pkl')