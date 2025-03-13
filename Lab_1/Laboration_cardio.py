import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class DiseasePrediction:
    def __init__(self):
        self.df = self.load_data()
        self.feature_engineer_bmi()
        self.feature_engineer_bp()
       
        
        


    def perform_eda(self):
        print(self.df.describe())
        print(self.df.info())
        print(self.df.value_counts("cardio")) # Visar antalet positiva & negativa med hjärt och kärlsjukdom
        self.piechart_cholesterol()
        self.plot_age_distribution()
        self.number_of_smokers()
        self.weight_distribution()
        self.height_distribution()
        self.gender_cardio()


    def load_data(self):
        df = pd.read_csv(r"C:\Code\ML-Tobias-Oberg-AI24\Lab_1\cardio_train.csv", sep=";")
        df.dropna()

        df["gender"] = df["gender"].map({1: "female", 2: "male"})
        df["cholesterol"] = df["cholesterol"].map({1: "normal", 2: "above normal", 3: "well above normal"})
        df["cardio"] = df["cardio"].map({0: "Negative", 1: "Positive"})
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
        return self.df





    def feature_engineer_bp(self): 
        Q1 = self.df[["ap_hi", "ap_lo"]].quantile(0.25) 
        Q3 = self.df[["ap_hi", "ap_lo"]].quantile(0.75) 
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = ~((self.df[["ap_hi", "ap_lo"]] < lower_bound) | (self.df[["ap_hi", "ap_lo"]] > upper_bound)).any(axis=1)
            
        self.df = self.df[mask]
            
        # Ville testa en annan metod för att ta bort outliers, svårt att avgöra gränserna själv.
        # Använde mig av samma metod i E03 logistic_regression.
        # Källa: https://www.geeksforgeeks.org/how-to-use-pandas-filter-with-iqr/
        
        
        
        def categorize_bp(ap_hi, ap_lo):
            if ap_hi < 90 or ap_lo < 60:
                return "Low Blood Pressure"
            elif 90 <= ap_hi <= 120 and 60 <= ap_lo <= 80:
                return "Normal Blood Pressure"
            elif 120 <= ap_hi <= 129 and ap_lo < 80:
                return "Elevated Blood Pressure"
            elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
                return "Hypertension Stage 1"
            elif 140 <= ap_hi <= 179 or 90 <= ap_lo <= 119:
                return "Hypertension Stage 2"
            elif ap_hi >= 180 or ap_lo >= 120:
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


        def add_percentage_labels(ax, data): # ChatGPT4, prompt: Hjälp mig skapa procent direkt på varje bar. Fick denna funktionen som respons. Jag ville ha mer tydlighet i subplotsen.
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



    def correlation_heat_map(self):
        plt.figure(figsize=(10,5))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True)
        plt.show()


    def copy_of_df(self):
        self.df1 = self.df.copy()
        self.df1 = self.df.drop(columns=["ap_hi", "ap_lo", "height", "weight", "BMI"])
        self.df1 = pd.get_dummies(self.df1, columns=["BMI_category", "Blood_pressure_category", "gender"],drop_first=True, dtype=int)


        self.df2 = self.df.copy()
        self.df2 = self.df.drop(columns=["BMI_category", "Blood_pressure_category", "height", "weight"])
        self.df2 = pd.get_dummies(self.df2, columns=["gender"], drop_first=True, dtype=int)

        print(self.df1.head())
        print(self.df2.head())
    # skapa kopia av dataset här och utför one-hot encoding samt släng vissa kolumner




    

    def confusion_matrix():
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        
        cn = confusion_matrix() # sätt in y_test och y_pred här
        ConfusionMatrixDisplay(cn).plot()
        
        return cn
        



    def classification_report():
        from sklearn.metrics import classification_report
        
        return print(classification_report()) # sätt in y_test och y_pred här
    