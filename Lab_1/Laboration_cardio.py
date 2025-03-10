import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class DiseasePrediction:
    def __init__(self):
        self.df = self.load_data()


    def perform_eda(self):
        self.EDA()
        self.piechart_cholesterol()
        self.plot_age_distribution()
        self.number_of_smokers()
        self.weight_distribution()
        self.height_distribution()
        self.gender_cardio()


    def load_data(self):
        df = pd.read_csv(r"C:\Code\ML-Tobias-Oberg-AI24\Lab_1\cardio_train.csv", sep=";")
        df.dropna()

        df["gender"] = df["gender"].replace({1: "female", 2: "male"})
        df["cholesterol"] = df["cholesterol"].replace({1: "normal", 2: "above normal", 3: "well above normal"})
        df["cardio"] = df["cardio"].replace({0: "Negative", 1: "Positive"})
        df["age"] = (df["age"] / 365).astype(int) # gör om age till år istället för dagar, int för hela år

        return df


    def EDA(self):
        print(self.df.describe())
        print(self.df.info())
        print(self.df.value_counts("cardio")) # Visar antalet positiva & negativa med hjärt och kärlsjukdom


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
        self.df["height_meters"] = self.df["height"] / 100    # för att omvandla cm till meter

        self.df["BMI"] = self.df["weight"] / (self.df["height_meters"] **2) # weight / height x height

        # lägg till något som tar bort outliers för BMI här, vart går gränsen för outliers eller extremvärden?


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

        self.df["BMI_category"] = self.df["BMI"].apply(categorize_bmi)
        return self.df





    def feature_engineer_bp():
        pass


    def visualize_diseases():
        pass # subplots



    def correlation_heat_map():
        pass


    def confusion_matrix():
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        
        cn = confusion_matrix() # sätt in y_test och y_pred här
        ConfusionMatrixDisplay(cn).plot()
        
        return cn
        



    def classification_report():
        from sklearn.metrics import classification_report
        
        return print(classification_report()) # sätt in y_test och y_pred här
    