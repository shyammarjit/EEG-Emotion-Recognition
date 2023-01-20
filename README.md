# EEG-Based Emotion Recognition Using Genetic Algorithm Optimized Multi-Layer Perceptron (IRIA 2021)

[Shyam Marjit](shyammarjit.github.io), [Upasana Talukdar](https://www.iiitg.ac.in/faculty/upasana/), and [Shyamanta M Hazarika](https://www.iitg.ac.in/s.m.hazarika/)

[![paper](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/abstract/document/9588702)
[![code](https://img.shields.io/badge/code-80:20-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B80-20%5D.ipynb)
[![code](https://img.shields.io/badge/code-10--fold-orange)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/%5BS01%5D%20%5BGA-MLP%5D%20%5B10-fold%5D.ipynb)
[![result](https://img.shields.io/badge/result-80:20-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/80-20%20GA-MLP%20results.md)
[![result](https://img.shields.io/badge/result-10--fold-blue)](https://github.com/shyammarjit/EEG-Emotion-Recognition/blob/IRIA-2021/10-fold%20GA-MLP%20results.md)
<hr />

> ***Abstract:*** Emotion Recognition is an important problemwithin Affective Computing and Human Computer Interaction. In recent years, various machine learning models have provided significant progress in the field of emotion recognition. This paper proposes a framework for EEG-based emotion recognition using Multi Layer Perceptron (MLP). Power Spectral Density features were used for quantifying the emotions interms of valence-arousal scale and MLP is used for classification. Genetic algorithm is used to optimize the architecture of MLP. The proposed model identifies a. two classes of emotions viz. Low/High Valence with an average accuracy of 91.10% and Low/High Arousal with an average accuracy of 91.02%, b. four classes of emotions viz. High Valence-Low Arousal (HVLA), High Valence-High Arousal (HVHA), Low Valence-Low Arousal (LVLA) and Low Valence-High Arousal (LVHA) with 83.52% accuracy. The reported results are better compared to existing results in the literature.<br/>
> ***Index Terms*** — EEG, Emotions, Power Spectral Density, Multi-Layer Perceptron, Genetic Algorithm.
<hr />


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
    <td style="text-align:center">92.2 &plusmn 8.84
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
    title={EEG-Based Emotion Recognition Using Genetic Algorithm Optimized Multi-Layer Perceptron},
    author={Marjit, Shyam and Talukdar, Upasana and Hazarika, Shyamanta M},
    booktitle={2021 International Symposium of Asian Control Association on Intelligent Robotics and Industrial Automation (IRIA)},
    year={2021},
    pages={304-309},
    organization={IEEE},
    doi={10.1109/IRIA53009.2021.9588702}
}
```
