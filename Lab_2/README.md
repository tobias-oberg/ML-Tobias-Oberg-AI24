# Körning av applikationen
Länk till webapplikation: 
skriv in streamlit run *filnamn.py* i terminalen för att köra den lokalt. 
https://filmrecommender.streamlit.app/ # länk till deployade applikationen.



## Metoder
Jag använder TFIDF-vektorisering(Term Frequency-Inverse Document Frequency) för att göra om text till numeriska värden.
Med hjälp av stopwords = "english" så tas vanliga ord bort som inte tillför semantisk information så som "the" och "i" i engelska språket. Bigrams(n_grams(1,2)) används för att få med två ord som är sammanhörande. Detta gör att ord som till exempel "Science Fiction" analyseras som en fras och inte två separata ord. TFIDF mäter både hur ofta ett ord förekommer men även hur unikt det är i hela datasetet. På detta sätt prioriteras termer som är representativa snarare än vanliga. 

För att jämföra vektorerna som skapas med hjälp av TFIDF så använder jag cosine similarity. Cosine similarity mäter hur lika två filmer är. Alltså, ju högre similaritet desto mer lika är filmerna. De 5 filmer med högst similaritet returneras. Denna metod mäter vinkeln mellan två vektorer i ett flerdimensionellt rum. Resultatet av detta är att det ger ett mått på hur lika de är oavsett längden på vektorerna. Ett värde närmare 1 innebär högre likhet. 


### Begränsningar
Applikationen använder ett dataset för filmer och ett annat för taggar. Om information saknas, så som taggar eller om datasetet innehåller felaktig data så gör detta att rekommendationerna har mindre träffsäkerhet. 

TF-IDF beräknar likheten mellan genrer och taggar. Om det inte finns tillräckligt beskrivande taggar eller genrer så uppstår en begränsning. Beräkningen tar inte hänsyn till andra faktorer som kan spela stor roll, till exempel skådespelare eller regissör som bara förekommer i vissa fall. 

En annan begränsning som sker är att det inte finns någon beskrivning om filmerna, detta leder till att många filmer klassificeras som liknande filmer på grund av likheten mellan genrer och taggar. 


#### Val
Jag använder pandas för datahantering, numpy för numeriska beräkningar, scikit-learn för tfidf och cosine similarity och streamlit för att köra applikationen på en websida då jag ville testa något nytt som liknar en dash-applikation. 

Denna applikationen använder en maskininlärningsteknik som är vanlig inom NLP och behöver inte tränas med labels. Det är content-based filtering med fokus på likhetsmått mellan texter och då valdes TFIDF och Cosine similarity. euclidian distance testades men upplevdes sämre och mindre robust och är även känsligare för variansen av längderna på vektorerna. 
Word2Vec och BERT kändes onödigt komplexa för detta problem samt att resultatet inte skulle göra en markant skillnad.  
Med TFIDF, jämfört med bag-of-words så är vektorerna mer meningsfulla. Istället för att alla ord ska vägas lika så får mer unika termer högre vikt. 

Att titta på genrer och taggar separat gav mindre relevanta förslag/resultat. Att inkludera titel testades men resultatet påverkades negativt. Det kan bero på att många titlar inte har semantisk information -- alltså en films titel beskriver oftast inte filmens genre till exempel, vilket leder till brus snarare än att vara till mer nytta. Dessutom är många titlar unika.

Att kombinera metadatan från tags och genrerna från movies antyder att de två features tillsammans fångar likheten på ett djupare plan. Genrer skapar en övergripande kategorisering och tags fångar mer detaljerade teman. 





