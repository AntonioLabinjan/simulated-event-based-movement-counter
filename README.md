Simulated event based movement time counter with tunable parameters

PARAMS EXPLAINED BY GPT:
---

**Pixel Intensity Threshold**

* Minimalna razlika u jačini piksela da bi se promatrala kao *event* (pokret).
* Niže vrijednosti → osjetljivije na male promjene (i šum).
* Veće vrijednosti → hvata samo jače pokrete.

---

**Pixel Change Area (sum of pixels)**

* Ukupna količina promijenjenih piksela potrebna da bi se pokret uopće priznao.
* Niže vrijednosti → svaki mali trzaj registrira kao pokret.
* Veće vrijednosti → ignorira sitne promjene (npr. treptaje, noise).

---

**Processing width (px)**

* Širina na koju se frame skalira za obradu (visina se računa proporcionalno).
* Manja širina → brže računanje, ali gubi se detalj.
* Veća širina → više detalja, ali CPU radi više.

---

**Membrane decay (alpha)**

* Koliko brzo neuron “zaboravlja” stari input.
* Niže vrijednosti (\~0.80) → jako curenje, neuroni brzo resetiraju napon.
* Više vrijednosti (\~0.99) → dugotrajnije pamćenje, kumulira impulse kroz više frameova.

---

**Layer1 spike threshold**

* Koliko pobude (nakon konvolucije) neuron u prvom sloju treba da bi *opalio spike*.
* Niže vrijednosti → češće spike-ovi (osjetljivije).
* Više vrijednosti → samo jaki eventi probijaju threshold.

---

**Layer2 spike threshold**

* Isto kao gore, ali za drugi sloj (koji već prima filtrirane spikeove iz Layer1).
* Radi kao “viša instanca” – skuplja spikeove iz L1 i odlučuje kad je događaj značajan.
* Niže vrijednosti → layer2 stalno puca.
* Više vrijednosti → layer2 puca samo kad se nakupe jaki eventi.

---

**Refractory frames (per neuron)**

* Broj frameova tijekom kojih neuron ne može ponovno pucati nakon što je spikeao.
* Veće vrijednosti → “hladi” neuron duže, sprječava prečesto pucanje.
* Manje vrijednosti (čak 0) → neuron može stalno pucati ako ima input.

---

**Trigger when final spikes >**

* Koliko ukupnih spikeova u Layer2 treba da se smatra da se dogodio *veliki event*.
* Ako finalni spike count prijeđe ovu vrijednost → aktivira se TRIGGER.
* Niže vrijednosti → lako se trigerira.
* Veće vrijednosti → treba baš masivan event za trigger.

---

**Layer1 kernel**

* Tip filtera za prvi sloj (na event maski).
* **sobel** → hvata rubove i pokrete u smjerovima.
* **gaussian** → lagano zamuti input, daje “mekši” signal.
* **box** → jednostavni prosjek, više noise-friendly ali grublji.

---

**Layer2 kernel**

* Isto, ali za drugi sloj.
* **box** → smanji spikeove šumom (pooling stil).
* **gaussian** → još glađe, više nalik biološkom neuronu koji integrira susjedne spikeove.

---
