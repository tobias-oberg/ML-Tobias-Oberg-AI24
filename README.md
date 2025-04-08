# Körning av applikationen
Länk till webapplikation: 
skriv in streamlit run *filnamn.py* i terminalen för att köra den lokalt. 



## Metoder
Jag använder TFIDF-vektorisering(Term Frequency-Inverse Document Frequency) för att göra om text till numeriska värden.
Med hjälp av stopwords = "english" så tas vanliga ord bort så som "the" och "i" i engelska språket. Bigrams(n_grams(1,2)) används för att få med två ord som är sammanhörande. Detta gör att ord som till exempel "Science Fiction" analyseras som en fras och inte två separata ord. 

För att jämföra vektorerna som skapas med hjälp av TFIDF så använder jag cosine similarity. Cosine similarity mäter hur lika två filmer är. Alltså, ju högre similaritet desto mer lika är filmerna. De 5 filmer med högst similaritet returneras. 


### Begränsningar
Applikationen använder ett dataset för filmer och ett annat för taggar. Om information saknas, så som taggar eller om datasetet innehåller felaktig data så gör detta att rekommendationerna har mindre träffsäkerhet. 

TF-IDF beräknar likheten mellan genrer och taggar. Om det inte finns tillräckligt beskrivande taggar eller genrer så uppstår en begränsning. Beräkningen tar inte hänsyn till andra faktorer som kan spela stor roll, till exempel skådespelare eller regissör som bara förekommer i vissa fall. 

En annan begränsning som sker är att det inte finns någon beskrivning om filmerna, detta leder till att många filmer klassificeras som liknande filmer på grund av likheten mellan genrer och taggar. 


#### Val
Jag använder pandas för datahantering, numpy för numeriska beräkningar, scikit-learn för tfidf och cosine similarity och streamlit för att köra applikationen på en websida då jag ville testa något nytt som liknar en dash-applikation. 
TFIDF och Cosine similarity valdes då den fick bäst resultat, euclidian distance testades men upplevdes sämre. 
Att slå ihop genrer och taggar fungerade bättre än att kolla på båda separat. Att lägga till titel testades men resultatet påverkades för mycket. 





