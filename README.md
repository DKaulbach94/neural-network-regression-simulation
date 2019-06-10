# neural-network-regression-simulation

This simulation was part of my master thesis.
By using the function Simulation() you can simulate a regression by a neural network
with different hyperparameters.
The main goal in this project was to understand the path which is taken in the high dimensional
parameter space of the neural network during the learning process.

• name: Ist ein String, der die angefertigten Graphen beschriftet. Es
soll die Lernmethode eingesetzt werden, zum Beispiel name = ”Stochastischer
Gradientenabstieg“.
• opt type: Ist ein Tensorflow-train-Objekt. Die Eingabe von ”opt type“
bestimmt, welches Lernverfahren verwendet wird.
• learning rate: Ist vom Typ float. Bestimmt die globale Lernrate.
• batch size: Ist vom Typ integer. Bestimmt, wie viele Trainingsdaten
erzeugt werden.
• minibatch size: Ist vom Typ integer. Bestimmt, wie groß die minibatches
sind, in die die Trainingsdaten eingeteilt werden.
• test size: Ist vom Typ integer. Bestimmt, aus wie vielen Daten jeweils
das empirische Risiko gesch¨atzt wird.
• ntests: Ist vom Typ integer. Bestimmt, wie oft insgesamt, in regelm
¨aßigen Abst¨anden w¨ahrend des Lernverfahrens, die Werte gemessen
werden, die bei der Ausgabe der Funktion aufgelistet sind.
• stddev: Ist vom Typ float. Legt die Standardabweichung der Output-
Daten fest, also entspricht dem Wert .
• diminish: Ist vom Typ boolean. Bei der Eingabe ”False“ wird der stochastische
Gradientenabstieg mit konstanter Lernrate durchgef¨uhrt.
Bei der Eingabe ”True“ nimmt die Lernrate mit jedem Schritt ab.
59
• distnr: Ist vom Typ integer und sollte zwischen 1 und ”ntests“ liegen.
”distnr“ markiert den Parametervektor ~ zum Zeitpunkt der Messung
der Nummer ”distnr“. Anschließend wird zu jedem weiteren Messzeitpunkt
der Abstand des aktuellen Parametervektors zu dem gespeicherten
Parametervektor gemessen.
