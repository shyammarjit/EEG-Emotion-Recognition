# EEG-Based Emotion Recognition Using Genetic Algorithm Optimized Multi-Layer Perceptron
***Abstract:*** Emotion Recognition is an important problemwithin Affective Computing and Human Computer Interaction.In recent years, various machine learning models have providedsignificant progress in the field of emotion recognition. Thispaper proposes a framework for EEG-based emotion recog-nition using Multi Layer Perceptron (MLP). Power SpectralDensity features were used for quantifying the emotions interms of valence-arousal scale and MLP is used for classification.Genetic algorithm is used to optimize the architecture of MLP.The proposed model identifies a. two classes of emotions viz. Low/High Valence with an average accuracy of 91.10% andLow/High Arousal with an average accuracy of 91.02%, b. fourclasses of emotions viz. High Valence-Low Arousal (HVLA), High Valence-High Arousal (HVHA), Low Valence-Low Arousal (LVLA) and Low Valence-High Arousal (LVHA) with 83.52%accuracy.The reported results are better compared to existing results in the literature.<br/>
***Index Terms*** — EEG, Emotions, Power Spectral Density, Multi-Layer Perceptron, Genetic Algorithm.

## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@INPROCEEDINGS{shyam2021eeg,
    title={EEG-Based Emotion Recognition Using Genetic Algorithm Optimized Multi-Layer Perceptron},
    author={Marjit, Shyam and Talukdar, Upasana and Hazarika, Shyamanta M},
    booktitle={2021 International Symposium of Asian Control Association on Intelligent Robotics and Industrial Automation (IRIA)},
    year={2021},
    pages={304-309},
    organization={IEEE},
    doi={10.1109/IRIA53009.2021.9588702}
}
```
paper [\[@IEEE Xplore]\](https://ieeexplore.ieee.org/abstract/document/9588702) [\[@researchgate]\](https://www.researchgate.net/profile/Shyam-Marjit-2/publication/355919224_EEG-Based_Emotion_Recognition_Using_Genetic_Algorithm_Optimized_Multi-Layer_Perceptron/links/61c1e26dabfb4634cb3361c9/EEG-Based-Emotion-Recognition-Using-Genetic-Algorithm-Optimized-Multi-Layer-Perceptron.pdf)<br/>
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
    <td style="text-align:center">1</td>
    <td style="text-align:center">2</td>
    <td style="text-align:center">3</td>
    <td style="text-align:center">4</td>
    <td style="text-align:center">5</td>
    <td style="text-align:center">6</td>
    <td style="text-align:center">7</td>
    <td style="text-align:center">8</td>
    <td style="text-align:center">9</td>
    </tr>
    </tbody>
    <tbody>
    <tr>
<td>10-fold cross validation</td>
    <td style="text-align:center">1</td>
    <td style="text-align:center">2</td>
    <td style="text-align:center">3</td>
    <td style="text-align:center">4</td>
    <td style="text-align:center">5</td>
    <td style="text-align:center">6</td>
    <td style="text-align:center">7</td>
    <td style="text-align:center">8</td>
    <td style="text-align:center">9</td>
</tr>
</tbody>
</table>
</div>
