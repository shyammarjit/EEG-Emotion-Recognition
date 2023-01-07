# EEG-Based Emotion Recognition Using Genetic Algorithm Optimized Multi-Layer Perceptron
***Abstract:*** Emotion Recognition is an important problemwithin Affective Computing and Human Computer Interaction.In recent years, various machine learning models have providedsignificant progress in the field of emotion recognition. Thispaper proposes a framework for EEG-based emotion recog-nition using Multi Layer Perceptron (MLP). Power SpectralDensity features were used for quantifying the emotions interms of valence-arousal scale and MLP is used for classification.Genetic algorithm is used to optimize the architecture of MLP.The proposed model identifies a. two classes of emotions viz. Low/High Valence with an average accuracy of 91.10% andLow/High Arousal with an average accuracy of 91.02%, b. fourclasses of emotions viz. High Valence-Low Arousal (HVLA), High Valence-High Arousal (HVHA), Low Valence-Low Arousal (LVLA) and Low Valence-High Arousal (LVHA) with 83.52%accuracy.The reported results are better compared to existing results in the literature.<br/>
***Index Terms*** — EEG, Emotions, Power Spectral Density, Multi-Layer Perceptron, Genetic Algorithm.

**Note for readers:**
The new version of code provides better accuracy due to the updated code for preprocesing step, rest all are same as it was descrived in the paper. The performance of the new version of code are noted in the below table:<br/>
<div class="block-language-tx"><table>
<caption id="prototypetable">TABLE-III: OVERALL PERFORMANCE OF GA-MLP CLASSIFIER (please refer paper)</caption>
<thead>
<tr>
<th></th>
<th style="text-align:center" colspan="3">Valence</th>
<th style="text-align:center" colspan="3">Arousal</th>
<th style="text-align:center" colspan="3">4-Types of emotions</th>
</tr>
<tr>
<th>Evolution Method</th>
<th style="text-align:center">Accuracy</th>
<th style="text-align:right">Precession</th>
<th style="text-align:right">Recall</th>
<th style="text-align:center">Accuracy</th>
<th style="text-align:right">Precession</th>
<th style="text-align:right">Recall</th>
<th style="text-align:center">Accuracy</th>
<th style="text-align:right">Precession</th>
<th style="text-align:right">Recall</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Randomly splitted training and <br/> testing with 80:20 partitions</td>
    <td style="text-align:center"><b>92.2 &plusmn 8.84<b>	
    <td style="text-align:center"><b>95.3 &plusmn 12<b>
    <td style="text-align:center"><b>84.5 &plusmn 22.1<b>
    <td style="text-align:center"><b> 92.97 &plusmn 8.95<b>
    <td style="text-align:center"><b> 93.18 &plusmn 11.82<b>
    <td style="text-align:center"><b> 91.4 &plusmn 17.1<b>
    <td style="text-align:center"><b> 85.94 &plusmn 13.75<b>
    <td style="text-align:center"><b> 88.58 &plusmn 12.43 <b>
    <td style="text-align:center"><b> 85.94 &plusmn 13.75 <b>
</tr>
    </tbody>
    <tbody>
    <tr>
<td>10-fold cross validation</td>
    <td style="text-align:center"><b> 96.64 &plusmn 2.43<b>	
    <td style="text-align:center"><b> 96.02 &plusmn 4.44<b>
    <td style="text-align:center"><b> 97.9 &plusmn 4.7<b>
    <td style="text-align:center"><b> 96.56 &plusmn 2.52<b>
    <td style="text-align:center"><b> 95.37 &plusmn 4.82<b>
    <td style="text-align:center"><b> 97.2 &plusmn 5.5<b>
    <td style="text-align:center"><b> 93.28 &plusmn 2.87<b>
    <td style="text-align:center"><b> 90.76 &plusmn 4.43<b>
    <td style="text-align:center"><b> 93.28 &plusmn 2.87<b>
</tr>
</tbody>
</table>
</div>

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@INPROCEEDINGS{shyam2021eeg,
    title={Enhancing {EEG-}Based Emotion Recognition using MultiDomain Features and Genetic Algorithm based Feature Selection},
    author={Marjit, Shyam and Talukdar, Upasana and Hazarika, Shyamanta M},
    booktitle={9th International Conference on Pattern Recognition and Machine Intelligence},
    year={2021},
    pages={-},
    organization={Springer},
    doi={}
}
```
[\[@paper\]](https://link.springer.com/book/9783031126994)<br/>
