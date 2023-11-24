from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression  # Replace with your model

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def index():
    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0
    intFeatures = ['Age']
    floatFeatures = []
    stringFeatures=['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

    train_df = pd.read_csv('survey.csv')
    
    # Dropping specific columns
    columns_to_drop = ['comments', 'state', 'Timestamp']
    train_df.drop(columns=columns_to_drop, inplace=True)

    # Handling missing data based on data types
    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df[feature] = train_df[feature].fillna(defaultFloat)

    # Cleaning 'Gender' column
    gender_iv = ['A little about you', 'p']
    train_df = train_df[~train_df['Gender'].isin(gender_iv)]

    # Filling missing age values with median
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    train_df.loc[train_df['Age'] < 18, 'Age'] = train_df['Age'].median()
    train_df.loc[train_df['Age'] > 120, 'Age'] = train_df['Age'].median()

    # Ranges of Age
    train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    # Replace NaN in 'self_employed' with 'No'
    train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

    # Replace NaN in 'work_interfere' with "Don't know"
    train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], "Don't know")
    

    # Encoding data
    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] = labelValue

    # Drop 'Country' column
    train_df = train_df.drop(['Country'], axis=1)
    new_data= new_data.drop(['Country'], axis=1)

    # Scaling 'Age' column using MinMaxScaler
    scaler = MinMaxScaler()
    train_df['Age'] = scaler.fit_transform(train_df[['Age']])
    

    # Defining X and y
    feature_cols1 = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols1]
    y = train_df.treatment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
   
         

     

    
    # Add your model training code here using X_train, y_train...

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])  # Convert to appropriate types
    gender = request.form['gender']
    country = request.form['country']
    self_employed = request.form['self_employed']
    family_history = request.form['family_history']
    treatment = request.form['treatment']
    work_interfere = request.form['work_interfere']
    no_employees = request.form['no_employees']
    remote_work = request.form['remote_work']
    tech_company = request.form['tech_company']
    benefits = request.form['benefits']
    care_options = request.form['care_options']
    wellness_program = request.form['wellness_program']
    seek_help = request.form['seek_help']
    anonymity = request.form['anonymity']
    leave = request.form['leave']
    mental_health_consequence = request.form['mental_health_consequence']
    phys_health_consequence = request.form['phys_health_consequence']
    coworkers = request.form['coworkers']
    supervisor = request.form['supervisor']
    mental_health_interview = request.form['mental_health_interview']
    phys_health_interview = request.form['phys_health_interview']
    mental_vs_physical = request.form['mental_vs_physical']
    new_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'Country': [country], 'self_employed': [self_employed],
        'family_history': [family_history], 'treatment': [treatment], 'work_interfere': [work_interfere],
        'no_employees': [no_employees], 'remote_work': [remote_work], 'tech_company': [tech_company],
        'benefits': [benefits], 'care_options': [care_options], 'wellness_program': [wellness_program],
        'seek_help': [seek_help], 'anonymity': [anonymity], 'leave': [leave],
        'mental_health_consequence': [mental_health_consequence], 'phys_health_consequence': [phys_health_consequence],
        'coworkers': [coworkers], 'supervisor': [supervisor],
        'mental_health_interview': [mental_health_interview], 'phys_health_interview': [phys_health_interview],
        'mental_vs_physical': [mental_vs_physical]
    })

    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0
    intFeatures = ['Age']
    floatFeatures = []
    stringFeatures=['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere', 'no_employees', 'remote_work', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview', 'mental_vs_physical', 'obs_consequence']

    train_df = pd.read_csv('survey.csv')
    
    # Dropping specific columns
    columns_to_drop = ['comments', 'state', 'Timestamp']
    train_df.drop(columns=columns_to_drop, inplace=True)

    # Handling missing data based on data types
    for feature in train_df:
        if feature in intFeatures:
            train_df[feature] = train_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
            train_df[feature] = train_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
            train_df[feature] = train_df[feature].fillna(defaultFloat)

    # Cleaning 'Gender' column
    gender_iv = ['A little about you', 'p']
    train_df = train_df[~train_df['Gender'].isin(gender_iv)]

    # Filling missing age values with median
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
    train_df.loc[train_df['Age'] < 18, 'Age'] = train_df['Age'].median()
    train_df.loc[train_df['Age'] > 120, 'Age'] = train_df['Age'].median()

    # Ranges of Age
    train_df['age_range'] = pd.cut(train_df['Age'], [0, 20, 30, 65, 100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

    # Replace NaN in 'self_employed' with 'No'
    train_df['self_employed'] = train_df['self_employed'].replace([defaultString], 'No')

    # Replace NaN in 'work_interfere' with "Don't know"
    train_df['work_interfere'] = train_df['work_interfere'].replace([defaultString], "Don't know")
    

    # Encoding data
    labelDict = {}
    for feature in train_df:
        le = preprocessing.LabelEncoder()
        le.fit(train_df[feature])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        train_df[feature] = le.transform(train_df[feature])
        # Get labels
        labelKey = 'label_' + feature
        labelValue = [*le_name_mapping]
        labelDict[labelKey] = labelValue

    # Drop 'Country' column
    train_df = train_df.drop(['Country'], axis=1)
    new_data= new_data.drop(['Country'], axis=1)

    # Scaling 'Age' column using MinMaxScaler
    scaler = MinMaxScaler()
    train_df['Age'] = scaler.fit_transform(train_df[['Age']])
    new_data_df['Age'] = scaler.fit_transform(new_data[['Age']])

    # Defining X and y
    feature_cols1 = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
    X = train_df[feature_cols1]
    y = train_df.treatment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
        
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    prediction= logreg.predict(new_data)
         

     

    
    # Add your model training code here using X_train, y_train...

    return render_template('result.html', prediction=prediction)
    
      # Include relevant model results or evaluation metrics



if __name__ == '__main__':
    app.run(debug=True)
