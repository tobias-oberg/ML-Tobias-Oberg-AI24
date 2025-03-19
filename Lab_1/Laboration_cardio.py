import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier


class DiseasePrediction:
    def __init__(self):
        self.df = self.load_data()
        
       
        

    # def perform_eda(self):
    #     # print(self.df.describe())
    #     # print(self.df.info())
    #     print(self.df.value_counts("cardio")) # Visar antalet positiva & negativa med hjärt och kärlsjukdom
    #     self.piechart_cholesterol()
    #     self.plot_age_distribution()
    #     self.number_of_smokers()
    #     self.weight_distribution()
    #     self.height_distribution()
    #     self.gender_cardio()


    def load_data(self):
        df = pd.read_csv(r"C:\Code\ML-Tobias-Oberg-AI24\Lab_1\cardio_train.csv", sep=";")
        df = df.dropna()

        df["gender"] = df["gender"].replace({1: "female", 2: "male"})
        df["cholesterol"] = df["cholesterol"].replace({1: "normal", 2: "above normal", 3: "well above normal"})
        df["cardio"] = df["cardio"].replace({0: "Negative", 1: "Positive"})
        df["age"] = (df["age"] / 365).astype(int) # gör om age till år istället för dagar, int för hela år

        return df


    def piechart_cholesterol(self):

        cholesterol_counts = self.df["cholesterol"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(cholesterol_counts.values, labels=cholesterol_counts.index, autopct="%1.1f%%")
        plt.title("Cholesterol levels")
        plt.show()


    def plot_age_distribution(self):
        plt.figure(figsize=(8,5))
        sns.histplot(data= self.df, x="age", hue="gender",kde=True, bins=30, multiple="dodge")
        plt.title("Age Distribution by Gender")
        plt.show()


    def number_of_smokers(self):
        plt.figure(figsize=(8,5))
        sns.histplot(data=self.df, x="age", hue="gender", bins=40, kde=True, multiple="dodge")
        plt.title("Distribution of smokers")

        number_of_smokers = self.df.groupby("smoke").size()
        smokers_percent = (number_of_smokers / len(self.df)) *100

        print(f"Number of smokers: {smokers_percent[1]:.2f}%")


    def weight_distribution(self):
        plt.figure(figsize=(8,5))
        sns.histplot(x="weight", data=self.df, hue="gender", bins=30, multiple="dodge")
        plt.xlabel("Weight (kg)")
        plt.title("Weight Distribution")
        plt.show()


    def height_distribution(self):
        plt.figure(figsize=(8,5))
        sns.histplot(x="height", data=self.df, hue="gender", bins=30, multiple="dodge")
        plt.xlabel("Height (cm)")
        plt.title("Height Distribution")
        plt.show()


    def gender_cardio(self):
        gender_totals = self.df.groupby("gender")["cardio"].count()
        df_cardio_positive = self.df[self.df["cardio"] == "Positive"].groupby("gender").size()
        df_cardio_positive_percentage = df_cardio_positive / gender_totals * 100

        fig, ax = plt.subplots()
        ax.pie(df_cardio_positive_percentage, labels=df_cardio_positive_percentage.index, autopct="%1.1f%%")
        plt.title("Positive cardio")
        plt.show()




    def feature_engineer_bmi(self):    # weight / height x height
        self.df["BMI"] = self.df["weight"] / ((self.df["height"] / 100) ** 2) # / 100 för att omvandla cm till meter
         # weight / height x height


        # BMI-värden tagna från WHO
        def categorize_bmi(BMI):
            if BMI < 18.5:
                return "Underweight"
            elif 18.5 <= BMI <= 24.9:
                return "Normal BMI"
            elif 25 <= BMI <= 29.9:
                return "Overweight"
            elif 30 <= BMI <= 34.9:
                return "Obese Class 1"
            elif 35 <= BMI <= 39.9:
                return "Obese Class 2"
            else:
                return "Obese Class 3"



        # filtrerar bort BMI under eller = 10, filtrerar bort BMI = 60 eller under
        self.df = self.df[(self.df["BMI"] >= 10) & (self.df["BMI"] <= 60)]
        self.df["BMI_category"] = self.df["BMI"].apply(categorize_bmi)
        



    def feature_engineer_bp(self): 
        Q1 = self.df[["ap_hi", "ap_lo"]].quantile(0.25) 
        Q3 = self.df[["ap_hi", "ap_lo"]].quantile(0.75) 
        IQR = Q3 - Q1

        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR

        mask = ~((self.df[["ap_hi", "ap_lo"]] < lower_bound) | (self.df[["ap_hi", "ap_lo"]] > upper_bound)).any(axis=1)
        print(f"Antal rader före filtrering: {self.df.shape[0]}")    
        self.df = self.df[mask]
        print(f"Antal rader efter filtrering: {self.df.shape[0]}")
            
        # Ville testa en annan metod för att ta bort outliers, svårt att avgöra gränserna själv.
        # Använde mig av samma metod i E03 logistic_regression.
        # Källa: https://www.geeksforgeeks.org/how-to-use-pandas-filter-with-iqr/
        
        
        
        def categorize_bp(ap_hi, ap_lo):
            if ap_hi < 90 or ap_lo < 60:
                return "Low Blood Pressure"
            elif ap_hi <= 120 and ap_lo <= 80:  # Normal BP
                return "Normal Blood Pressure"
            elif ap_hi <= 129 and ap_lo < 80:  # Elevated
                return "Elevated Blood Pressure"
            elif ap_hi <= 139 or ap_lo <= 89:  # Hypertension Stage 1
                return "Hypertension Stage 1"
            elif ap_hi <= 179 or ap_lo <= 119:  # Hypertension Stage 2
                return "Hypertension Stage 2"
            else:  # Om ap_hi >= 180 eller ap_lo >= 120
                return "Hypertensive Crisis"

            
        self.df["Blood_pressure_category"] = self.df.apply(lambda row: categorize_bp(row["ap_hi"], row["ap_lo"]), axis=1)
        return self.df



    def visualize_diseases(self):
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        self.df["cardio"] = self.df["cardio"].map({"Positive": 1, "Negative": 0}).astype(int) # omvandlar tillbaka "cardio" till numeriska värden så att jag kan köra .mean(). Inte den snyggaste lösningen.

        # Beräkna andelen positiva fall för olika kategorier
        blodtryck_andel = self.df.groupby("Blood_pressure_category")["cardio"].mean().reset_index()
        bmi_andel = self.df.groupby("BMI_category")["cardio"].mean().reset_index()
        kolesterol_andel = self.df.groupby("cholesterol")["cardio"].mean().reset_index()
        aktivitet_andel = self.df.groupby("active")["cardio"].mean().reset_index()


        def add_percentage_labels(ax, data): # ChatGPT-4o, prompt: Hjälp mig skapa procent direkt på varje bar. Fick denna funktionen som respons. Jag ville ha mer tydlighet i subplotsen.
            for p in ax.patches:
                height = p.get_height()
                percentage = height * 100
                ax.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{percentage:.1f}%', ha='center', va='bottom') #

      
        sns.barplot(x="Blood_pressure_category", y="cardio", data=blodtryck_andel, ax=axes[0, 0], hue="Blood_pressure_category", palette="viridis", legend=False)
        axes[0, 0].set_title("Positive cases by Blood Pressure Category")
        axes[0, 0].set_xlabel("Blood Pressure Category")
        axes[0, 0].set_ylabel("Proportion of Positive Cases")
        add_percentage_labels(axes[0, 0], blodtryck_andel)

        sns.barplot(x="BMI_category", y="cardio", data=bmi_andel, ax=axes[0, 1], hue="BMI_category", palette="magma", legend=False)
        axes[0, 1].set_title("Positive cases by BMI Category")
        axes[0, 1].set_xlabel("BMI Category")
        axes[0, 1].set_ylabel("Proportion of Positive Cases")
        add_percentage_labels(axes[0, 1], bmi_andel)
        

        sns.barplot(x="cholesterol", y="cardio", data=kolesterol_andel, ax=axes[1, 0], hue= "cholesterol",palette="plasma", legend=False)
        axes[1, 0].set_title("Positive cases by Cholesterol Level")
        axes[1, 0].set_xlabel("Cholesterol Level")
        axes[1, 0].set_ylabel("Proportion of Positive Cases")
        add_percentage_labels(axes[1, 0], kolesterol_andel)


        sns.barplot(x="active", y="cardio", data=aktivitet_andel, ax=axes[1, 1], hue="active", palette="cividis", legend=False)
        axes[1, 1].set_title("Positive cases by Activity Level")
        axes[1, 1].set_xlabel("Activity Level")
        axes[1, 1].set_ylabel("Proportion of Positive Cases")
        add_percentage_labels(axes[1, 1], aktivitet_andel)

        
        plt.tight_layout()
        plt.show()



    def correlation_heat_map(self): # Korrelationsmatris
        plt.figure(figsize=(10,5))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True)
        plt.show()


    def copy_of_df(self):
        self.df = pd.get_dummies(self.df, columns=["cholesterol"], drop_first=True)
        self.df1 = self.df.copy()
        self.df1 = self.df1.drop(columns=["ap_hi", "ap_lo", "height", "weight", "BMI"])
        self.df1 = pd.get_dummies(self.df1, columns=["BMI_category", "Blood_pressure_category", "gender"],drop_first=True, dtype=int)


        self.df2 = self.df.copy()
        self.df2 = self.df2.drop(columns=["BMI_category", "Blood_pressure_category", "height", "weight"])
        self.df2 = pd.get_dummies(self.df2, columns=["gender"], drop_first=True, dtype=int)

        # print(self.df1.head())
        # print(self.df2.head())
    # skapa kopia av dataset här och utför one-hot encoding samt släng vissa kolumner
        

class ModelTraining:

    def __init__(self, df):
        self.df = df
        self.best_models = {}
        self.best_score = {}
      
        
        
    def split_data(self, test_size=0.3, val_size=0.5): # Splittar datan i träningsdata, valideringsdata och testdata. 
        X = self.df.drop("cardio", axis=1)
        y = self.df["cardio"]
        print(self.df.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size)
        return X_train, X_val, X_test, y_train, y_val, y_test



    def scale_data(self, X_train, X_val, X_test):  
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        

        scaler = MinMaxScaler()
        X_train_norm = scaler.fit_transform(X_train_scaled)
        X_val_norm = scaler.transform(X_val_scaled)
        X_test_norm = scaler.transform(X_test_scaled)
            
        

        return X_train_norm, X_val_norm, X_test_norm

    

    def hyper_tuning(self, model_name, X_train, y_train): # Hyperparameter tuning för att hitta bästa modellen
        models = {
            "LogisticRegression": (LogisticRegression(), { # Hyperparametrar för Logistic Regression
                "C": [0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "saga"],
                "max_iter": [1000, 5000, 10000]
            }),
            "RandomForest": (RandomForestClassifier(), { # Hyperparametrar för Random Forest
                "n_estimators": [50, 100],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            }),
            "GradientBoosting": (GradientBoostingClassifier(), {  # Hyperparametrar för Gradient Boosting
                "n_estimators": [50, 100, 200],  
                "learning_rate": [0.01, 0.1, 0.2],  
                "max_depth": [3, 5, 7] 
            })
        }

       
        model, param_grid = models[model_name] # Hämtar modell och parametrar för specifik modell
        print(f"Hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1) # cv=3 för att dela upp datan i 3 delar. Prövade med cv=5 men det tog alldeles för lång tid.
        grid_search.fit(X_train, y_train) # Tränar modellen
        
        self.best_models[model_name] = grid_search.best_estimator_ # Sparar bästa modellen
        print(f"Best model for {model_name}: {grid_search.best_params_}\n")

    

    def train_model(self, model_name, X_train, y_train, X_val, y_val): # Tränar modellen
        model = self.best_models[model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)               # Predictar valideringsdatan
        accuracy = accuracy_score(y_val, y_pred)    # Beräknar accuracy
        self.best_score [model_name] = accuracy
        print(f"{model_name} - Validation accuracy: {accuracy:.4f}\n")

        return accuracy



    def voting_classifier(self, X_train, y_train, X_val, y_val): # Skapar en Voting Classifier
        estimators = [(name, model) for name, model in self.best_models.items()] # Hämtar de bästa modellerna
        voting_clf = VotingClassifier(estimators=estimators, voting='hard') 

         
        voting_clf.fit(X_train, y_train) 
        y_pred = voting_clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Voting Classifier - Validation accuracy: {accuracy:.4f}\n")


    def display_confusion_matrix_and_report(self, model_name, X_test, y_test): # confusion matrix och classification report
        model = self.best_models[model_name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred) 
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"Confusion Matrix model {model}") 
        print(classification_report(y_test, y_pred))
        return cm
        

    