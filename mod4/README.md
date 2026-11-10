# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py
sentiment
Epoch 1, loss 31.410427981870313, train accuracy: 52.89%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 2, loss 31.213534000866343, train accuracy: 50.00%
Validation accuracy: 49.00%
Best Valid accuracy: 49.00%
Epoch 3, loss 31.119839660327777, train accuracy: 51.56%
Validation accuracy: 57.00%
Best Valid accuracy: 57.00%
Epoch 4, loss 30.77405632307956, train accuracy: 56.67%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 5, loss 30.64344184167784, train accuracy: 57.56%
Validation accuracy: 62.00%
Best Valid accuracy: 63.00%
Epoch 6, loss 30.482841302906472, train accuracy: 55.78%
Validation accuracy: 49.00%
Best Valid accuracy: 63.00%
Epoch 7, loss 30.24906613888077, train accuracy: 57.33%
Validation accuracy: 54.00%
Best Valid accuracy: 63.00%
Epoch 8, loss 29.983092373193244, train accuracy: 61.33%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 9, loss 29.517363272473304, train accuracy: 63.56%
Validation accuracy: 58.00%
Best Valid accuracy: 66.00%
Epoch 10, loss 29.140598598378244, train accuracy: 66.22%
Validation accuracy: 61.00%
Best Valid accuracy: 66.00%
Epoch 11, loss 28.816042432904986, train accuracy: 66.00%
Validation accuracy: 61.00%
Best Valid accuracy: 66.00%
Epoch 12, loss 28.294147761231113, train accuracy: 66.22%
Validation accuracy: 63.00%
Best Valid accuracy: 66.00%
Epoch 13, loss 27.843383280671382, train accuracy: 66.67%
Validation accuracy: 64.00%
Best Valid accuracy: 66.00%
Epoch 14, loss 27.252346586655563, train accuracy: 70.44%
Validation accuracy: 61.00%
Best Valid accuracy: 66.00%
Epoch 15, loss 26.76015578259569, train accuracy: 71.78%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 16, loss 26.48142389150727, train accuracy: 68.89%
Validation accuracy: 63.00%
Best Valid accuracy: 66.00%
Epoch 17, loss 25.820309913200095, train accuracy: 74.00%
Validation accuracy: 65.00%
Best Valid accuracy: 66.00%
Epoch 18, loss 25.096495162218645, train accuracy: 73.78%
Validation accuracy: 62.00%
Best Valid accuracy: 66.00%
Epoch 19, loss 23.929308014364224, train accuracy: 77.78%
Validation accuracy: 64.00%
Best Valid accuracy: 66.00%
Epoch 20, loss 23.15196155288618, train accuracy: 75.33%
Validation accuracy: 62.00%
Best Valid accuracy: 66.00%
Epoch 21, loss 23.406267352818766, train accuracy: 74.67%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 22, loss 22.408276088732258, train accuracy: 77.11%
Validation accuracy: 66.00%
Best Valid accuracy: 66.00%
Epoch 23, loss 21.952315415866074, train accuracy: 75.33%
Validation accuracy: 72.00%
Best Valid accuracy: 72.00%
Epoch 24, loss 21.558531601605406, train accuracy: 78.44%
Validation accuracy: 65.00%
Best Valid accuracy: 72.00%
Epoch 25, loss 20.59651398468635, train accuracy: 79.33%
Validation accuracy: 67.00%
Best Valid accuracy: 72.00%
Epoch 26, loss 19.989250039808407, train accuracy: 81.33%
Validation accuracy: 65.00%
Best Valid accuracy: 72.00%
Epoch 27, loss 19.227823668612064, train accuracy: 81.11%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 28, loss 19.367012141890267, train accuracy: 80.44%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 29, loss 18.880970737623734, train accuracy: 79.78%
Validation accuracy: 62.00%
Best Valid accuracy: 72.00%
Epoch 30, loss 17.899696113171704, train accuracy: 81.56%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 31, loss 18.040001427345896, train accuracy: 81.33%
Validation accuracy: 65.00%
Best Valid accuracy: 72.00%
Epoch 32, loss 17.962980153161798, train accuracy: 79.33%
Validation accuracy: 69.00%
Best Valid accuracy: 72.00%
Epoch 33, loss 17.131931870020846, train accuracy: 80.89%
Validation accuracy: 71.00%
Best Valid accuracy: 72.00%
Epoch 34, loss 17.129141683329582, train accuracy: 80.89%
Validation accuracy: 69.00%
Best Valid accuracy: 72.00%
Epoch 35, loss 16.14438203371953, train accuracy: 84.00%
Validation accuracy: 64.00%
Best Valid accuracy: 72.00%
Epoch 36, loss 15.042166282144917, train accuracy: 86.67%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 37, loss 15.320101099175298, train accuracy: 83.56%
Validation accuracy: 70.00%
Best Valid accuracy: 72.00%
Epoch 38, loss 14.846545381954696, train accuracy: 84.89%
Validation accuracy: 65.00%
Best Valid accuracy: 72.00%
Epoch 39, loss 14.875531265506702, train accuracy: 86.22%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%

mnist

Epoch 1 loss 2.2872652048553928 valid acc 2/16
Epoch 1 loss 11.4794097991979 valid acc 3/16
Epoch 1 loss 11.516803736555502 valid acc 2/16
Epoch 1 loss 11.381265375040952 valid acc 2/16
Epoch 1 loss 11.434242708693642 valid acc 2/16
Epoch 1 loss 11.118980805789008 valid acc 4/16
Epoch 1 loss 10.879875232850729 valid acc 7/16
Epoch 1 loss 10.479571568192954 valid acc 8/16
Epoch 1 loss 9.503238403111443 valid acc 9/16
Epoch 1 loss 7.804900982759845 valid acc 8/16
Epoch 1 loss 7.807797176622733 valid acc 8/16
Epoch 1 loss 7.563256756668486 valid acc 8/16
Epoch 1 loss 7.351399241533265 valid acc 13/16
Epoch 1 loss 6.660553241976299 valid acc 9/16
Epoch 1 loss 6.899936416795751 valid acc 7/16
Epoch 1 loss 6.2389372282506805 valid acc 11/16
Epoch 1 loss 7.140512845751586 valid acc 12/16
Epoch 1 loss 6.005976857302007 valid acc 12/16
Epoch 1 loss 5.317192322526571 valid acc 13/16
Epoch 1 loss 5.635391393058814 valid acc 12/16
Epoch 1 loss 4.499764934357527 valid acc 11/16
Epoch 1 loss 4.583747534117902 valid acc 10/16
Epoch 1 loss 3.2544034616857753 valid acc 11/16
Epoch 1 loss 4.260627365461752 valid acc 11/16
Epoch 1 loss 4.150527724808045 valid acc 8/16
Epoch 1 loss 4.536365135008882 valid acc 13/16
Epoch 1 loss 5.25464145395126 valid acc 13/16
Epoch 1 loss 3.0960413272203318 valid acc 13/16
Epoch 1 loss 4.1271132662775605 valid acc 12/16
Epoch 1 loss 2.488843311731474 valid acc 14/16
Epoch 1 loss 4.446860131297981 valid acc 10/16
Epoch 1 loss 4.029225397501301 valid acc 11/16
Epoch 1 loss 3.8877366832302283 valid acc 13/16
Epoch 1 loss 3.787037187248316 valid acc 11/16
Epoch 1 loss 5.873055715816236 valid acc 12/16
Epoch 1 loss 4.198914961092875 valid acc 11/16
Epoch 1 loss 3.309376843015832 valid acc 13/16
Epoch 1 loss 3.146024709846948 valid acc 12/16
Epoch 1 loss 3.8409366166160304 valid acc 13/16
Epoch 1 loss 3.76802212069092 valid acc 13/16
Epoch 1 loss 3.3965906080365436 valid acc 10/16
Epoch 1 loss 3.8298287317268196 valid acc 13/16
Epoch 1 loss 3.504005669093799 valid acc 14/16
Epoch 1 loss 2.7032408061322606 valid acc 12/16
Epoch 1 loss 4.265121384288044 valid acc 13/16
Epoch 1 loss 2.9197978460213654 valid acc 13/16
Epoch 1 loss 3.1179185299732004 valid acc 14/16
Epoch 1 loss 2.9029339249278916 valid acc 15/16
Epoch 1 loss 2.1990989704874786 valid acc 13/16
Epoch 1 loss 2.114671375895099 valid acc 12/16
Epoch 1 loss 3.2858808361446585 valid acc 13/16
Epoch 1 loss 2.556503959278758 valid acc 15/16
Epoch 1 loss 3.643714844321361 valid acc 14/16
Epoch 1 loss 2.5915598749277304 valid acc 12/16
Epoch 1 loss 2.6673509906944544 valid acc 14/16
Epoch 1 loss 2.4089930371468666 valid acc 12/16
Epoch 1 loss 2.5739477602125076 valid acc 14/16
Epoch 1 loss 3.5093560445656955 valid acc 13/16
Epoch 1 loss 2.303309274723372 valid acc 13/16
Epoch 1 loss 2.793484741166022 valid acc 13/16
Epoch 1 loss 3.372974057404744 valid acc 14/16
Epoch 1 loss 2.3804431999967344 valid acc 14/16
Epoch 1 loss 3.8849465884148815 valid acc 14/16
Epoch 2 loss 0.488352696668855 valid acc 15/16
Epoch 2 loss 2.007714842340714 valid acc 14/16
Epoch 2 loss 2.833084419327646 valid acc 15/16
Epoch 2 loss 3.2151894247562094 valid acc 15/16
Epoch 2 loss 2.41468276382133 valid acc 15/16
Epoch 2 loss 2.201854697208721 valid acc 15/16
Epoch 2 loss 3.064876972902865 valid acc 13/16
Epoch 2 loss 3.5543487784142567 valid acc 14/16
Epoch 2 loss 3.7656887480329355 valid acc 13/16
Epoch 2 loss 2.034033156945486 valid acc 14/16
Epoch 2 loss 2.577805382396828 valid acc 13/16
Epoch 2 loss 3.2790906952836107 valid acc 15/16
Epoch 2 loss 2.6134543198722806 valid acc 14/16
Epoch 2 loss 2.452752094135996 valid acc 12/16
Epoch 2 loss 3.458420948207607 valid acc 14/16
Epoch 2 loss 2.67617145563859 valid acc 13/16
Epoch 2 loss 3.677375068696408 valid acc 13/16
Epoch 2 loss 2.7080756386492992 valid acc 14/16
Epoch 2 loss 2.3420629069306957 valid acc 12/16
Epoch 2 loss 2.65335873117922 valid acc 12/16
Epoch 2 loss 2.1623133163554726 valid acc 14/16
Epoch 2 loss 2.0871452131061274 valid acc 13/16
Epoch 2 loss 1.296098937547259 valid acc 14/16
Epoch 2 loss 3.2733024566592475 valid acc 13/16
Epoch 2 loss 1.9903053541099682 valid acc 13/16
Epoch 2 loss 1.8624188718436627 valid acc 14/16
Epoch 2 loss 2.6093753946405487 valid acc 15/16
Epoch 2 loss 2.9113871892175864 valid acc 15/16
Epoch 2 loss 1.845576905242996 valid acc 13/16
Epoch 2 loss 1.491295275890817 valid acc 13/16
Epoch 2 loss 2.3330061244603644 valid acc 14/16
Epoch 2 loss 1.9053446395173896 valid acc 13/16
Epoch 2 loss 1.1880817271850308 valid acc 14/16
Epoch 2 loss 2.3364646983044945 valid acc 16/16
Epoch 2 loss 2.4582594057076546 valid acc 15/16
Epoch 2 loss 2.452389222697675 valid acc 15/16
Epoch 2 loss 1.9545228042788276 valid acc 11/16
Epoch 2 loss 1.9480355513126975 valid acc 13/16
Epoch 2 loss 2.7204698012087034 valid acc 14/16
Epoch 2 loss 1.8853455303693625 valid acc 14/16
Epoch 2 loss 1.8034115039938503 valid acc 13/16
Epoch 2 loss 2.16626444326482 valid acc 16/16
Epoch 2 loss 1.9271313320931664 valid acc 14/16
Epoch 2 loss 1.7096077940068013 valid acc 15/16
Epoch 2 loss 3.0451822511322586 valid acc 15/16
Epoch 2 loss 1.5950958487363494 valid acc 15/16
Epoch 2 loss 2.1728870433770924 valid acc 14/16
Epoch 2 loss 1.9851838216728033 valid acc 16/16
Epoch 2 loss 1.6414011211902984 valid acc 15/16
Epoch 2 loss 1.4667301150562486 valid acc 15/16
Epoch 2 loss 2.029426190315002 valid acc 13/16
Epoch 2 loss 1.4423660259688202 valid acc 14/16
Epoch 2 loss 2.341501558669674 valid acc 15/16
Epoch 2 loss 1.0735978773112522 valid acc 16/16
Epoch 2 loss 1.9681834140189598 valid acc 13/16
Epoch 2 loss 1.171362410286683 valid acc 14/16
Epoch 2 loss 1.4772990652551092 valid acc 15/16
Epoch 2 loss 1.999312358362869 valid acc 14/16
Epoch 2 loss 2.2906111161237694 valid acc 13/16
Epoch 2 loss 1.5294477361575558 valid acc 15/16
Epoch 2 loss 1.7095456010185994 valid acc 13/16
Epoch 2 loss 1.7814073646744588 valid acc 15/16
Epoch 2 loss 2.0107928623812086 valid acc 14/16
Epoch 3 loss 0.3008956116032032 valid acc 16/16
Epoch 3 loss 1.770863837796569 valid acc 15/16
Epoch 3 loss 2.4725837667848953 valid acc 16/16
Epoch 3 loss 2.1109892967353683 valid acc 14/16
Epoch 3 loss 1.5755969603820401 valid acc 15/16
Epoch 3 loss 1.9652069940841321 valid acc 15/16
Epoch 3 loss 2.6733247270701055 valid acc 14/16
Epoch 3 loss 2.204198537418598 valid acc 13/16
Epoch 3 loss 1.8379138848698806 valid acc 14/16
Epoch 3 loss 2.1185515353537743 valid acc 15/16
Epoch 3 loss 1.4082213383376745 valid acc 14/16
Epoch 3 loss 2.693174985479481 valid acc 14/16
Epoch 3 loss 2.292737955343444 valid acc 15/16
Epoch 3 loss 2.0943530539798445 valid acc 14/16
Epoch 3 loss 2.2594956331817575 valid acc 13/16
Epoch 3 loss 1.7614110954823958 valid acc 16/16
Epoch 3 loss 3.4909796905330897 valid acc 14/16
Epoch 3 loss 2.318237563245612 valid acc 14/16
Epoch 3 loss 1.7866807638974342 valid acc 15/16
Epoch 3 loss 1.6702577497773148 valid acc 15/16
Epoch 3 loss 2.0279265493110636 valid acc 13/16
Epoch 3 loss 1.7834517543995774 valid acc 14/16
Epoch 3 loss 0.7139208551568578 valid acc 15/16
Epoch 3 loss 1.4772871953595834 valid acc 14/16
Epoch 3 loss 1.5500229541789392 valid acc 15/16
Epoch 3 loss 1.5670514481052855 valid acc 15/16
Epoch 3 loss 1.4162346224165687 valid acc 14/16
Epoch 3 loss 1.802842259407244 valid acc 14/16
Epoch 3 loss 1.4817597109231857 valid acc 14/16
Epoch 3 loss 0.7313920934315775 valid acc 15/16
Epoch 3 loss 1.584533479312079 valid acc 15/16
Epoch 3 loss 1.2035579710104918 valid acc 13/16
Epoch 3 loss 0.6207483078661542 valid acc 13/16
Epoch 3 loss 1.612667778695474 valid acc 15/16
Epoch 3 loss 2.0702888667814365 valid acc 15/16
Epoch 3 loss 1.4687916578143017 valid acc 15/16
Epoch 3 loss 1.7288687818496298 valid acc 16/16
Epoch 3 loss 1.570457211088764 valid acc 15/16
Epoch 3 loss 1.668474622294391 valid acc 15/16
Epoch 3 loss 1.2018055418519755 valid acc 14/16
Epoch 3 loss 1.2899680343354936 valid acc 16/16
Epoch 3 loss 1.472932511400439 valid acc 14/16
Epoch 3 loss 1.1627498369470266 valid acc 16/16
Epoch 3 loss 1.2742619534280306 valid acc 13/16
Epoch 3 loss 2.0243287587752192 valid acc 14/16
Epoch 3 loss 0.7571815387377393 valid acc 16/16
Epoch 3 loss 1.8554318254370974 valid acc 15/16
Epoch 3 loss 2.032634623839869 valid acc 14/16
Epoch 3 loss 1.1843146752768472 valid acc 14/16
Epoch 3 loss 1.4048032307007499 valid acc 14/16
Epoch 3 loss 0.7897960084039711 valid acc 15/16
Epoch 3 loss 1.2345476177974462 valid acc 15/16
Epoch 3 loss 1.8468689287617719 valid acc 15/16
Epoch 3 loss 1.18036200009332 valid acc 16/16
Epoch 3 loss 1.515209702523419 valid acc 15/16
Epoch 3 loss 0.9954584623094972 valid acc 15/16
Epoch 3 loss 1.1958840643324247 valid acc 15/16
Epoch 3 loss 1.557003132706578 valid acc 13/16
Epoch 3 loss 1.4450028821140484 valid acc 15/16
Epoch 3 loss 1.4443251909643837 valid acc 12/16
Epoch 3 loss 2.367049189948327 valid acc 15/16
Epoch 3 loss 0.9580013147798428 valid acc 14/16
Epoch 3 loss 1.8507328675360852 valid acc 15/16
Epoch 4 loss 0.10324763870385835 valid acc 15/16
Epoch 4 loss 1.2108759967015266 valid acc 16/16
Epoch 4 loss 2.016770058624171 valid acc 16/16
Epoch 4 loss 1.5507382479648413 valid acc 16/16
Epoch 4 loss 1.207038344287997 valid acc 15/16
Epoch 4 loss 0.9519483240421318 valid acc 15/16
Epoch 4 loss 1.805999120471119 valid acc 15/16
Epoch 4 loss 1.898005724609955 valid acc 16/16
Epoch 4 loss 1.6172308031895348 valid acc 14/16
Epoch 4 loss 1.4623018141293664 valid acc 15/16
Epoch 4 loss 1.2810857429297307 valid acc 15/16
Epoch 4 loss 3.008036729175743 valid acc 16/16
Epoch 4 loss 1.5215014670220652 valid acc 13/16
Epoch 4 loss 2.2633497592434946 valid acc 14/16
Epoch 4 loss 2.2327673009852815 valid acc 14/16
Epoch 4 loss 1.0839681548953297 valid acc 13/16
Epoch 4 loss 2.1452647212909697 valid acc 15/16
Epoch 4 loss 1.7297089842429045 valid acc 14/16
Epoch 4 loss 2.095876549349561 valid acc 14/16
Epoch 4 loss 0.9447333744508022 valid acc 14/16
Epoch 4 loss 1.8051760817038376 valid acc 14/16
Epoch 4 loss 1.0980850658612435 valid acc 14/16
Epoch 4 loss 0.5810969058259386 valid acc 15/16
Epoch 4 loss 1.412650484191801 valid acc 14/16
Epoch 4 loss 0.6515100549768965 valid acc 13/16
Epoch 4 loss 1.6211163507293986 valid acc 15/16
Epoch 4 loss 1.0059041667100714 valid acc 16/16
Epoch 4 loss 0.7959302116642795 valid acc 14/16
Epoch 4 loss 1.0742063232410213 valid acc 14/16
Epoch 4 loss 0.9430174964734714 valid acc 14/16
Epoch 4 loss 1.5365110354033014 valid acc 16/16
Epoch 4 loss 0.8390637486449126 valid acc 15/16
Epoch 4 loss 0.6286521884926591 valid acc 15/16
Epoch 4 loss 1.5973817079194066 valid acc 15/16
Epoch 4 loss 1.5745696944208727 valid acc 16/16
Epoch 4 loss 1.3137629169812741 valid acc 15/16
Epoch 4 loss 1.723881392105759 valid acc 15/16
Epoch 4 loss 0.990428168779281 valid acc 15/16
Epoch 4 loss 1.749305853485458 valid acc 16/16
Epoch 4 loss 1.128660293483263 valid acc 16/16
Epoch 4 loss 0.37063359153714315 valid acc 14/16
Epoch 4 loss 0.9370299137015787 valid acc 16/16
Epoch 4 loss 1.2164536497918932 valid acc 14/16
Epoch 4 loss 0.905438840359319 valid acc 14/16
Epoch 4 loss 2.0947480197341997 valid acc 16/16
Epoch 4 loss 0.7684532659495509 valid acc 15/16
Epoch 4 loss 1.1007801792333187 valid acc 15/16
Epoch 4 loss 1.3388903986472762 valid acc 16/16
Epoch 4 loss 0.7755316672483955 valid acc 16/16
Epoch 4 loss 0.6875919128704967 valid acc 16/16
Epoch 4 loss 0.8582218608947507 valid acc 15/16
Epoch 4 loss 1.0597302205406636 valid acc 16/16
Epoch 4 loss 1.7188698845556707 valid acc 14/16
Epoch 4 loss 0.6232958184933625 valid acc 14/16
Epoch 4 loss 1.3717357461115065 valid acc 15/16
Epoch 4 loss 0.748676833335781 valid acc 16/16
Epoch 4 loss 0.5017077784249149 valid acc 15/16
Epoch 4 loss 1.2051456478502829 valid acc 16/16
Epoch 4 loss 1.227110302277477 valid acc 14/16
Epoch 4 loss 1.3791915343132606 valid acc 14/16
Epoch 4 loss 1.4768672483357566 valid acc 14/16
Epoch 4 loss 1.2823227188405006 valid acc 14/16
Epoch 4 loss 1.2548421656296622 valid acc 15/16
Epoch 5 loss 0.080005925978518 valid acc 16/16
Epoch 5 loss 1.0674703035205126 valid acc 16/16
Epoch 5 loss 1.3726125124773318 valid acc 16/16
Epoch 5 loss 1.3310450180087696 valid acc 14/16
Epoch 5 loss 0.7847716766710928 valid acc 14/16
Epoch 5 loss 1.0388555320041726 valid acc 16/16
Epoch 5 loss 2.092864111075012 valid acc 15/16
Epoch 5 loss 1.1344827293764042 valid acc 15/16
Epoch 5 loss 1.2283116477102614 valid acc 16/16
Epoch 5 loss 1.0712049680326188 valid acc 15/16
Epoch 5 loss 1.06295818657336 valid acc 15/16
Epoch 5 loss 1.722492489447829 valid acc 15/16
Epoch 5 loss 1.5501072309600292 valid acc 15/16
Epoch 5 loss 1.4848732246270009 valid acc 14/16
Epoch 5 loss 2.074221579796359 valid acc 15/16
Epoch 5 loss 0.8656794696155747 valid acc 15/16
Epoch 5 loss 2.341724322265452 valid acc 16/16
Epoch 5 loss 1.710243662332308 valid acc 15/16
Epoch 5 loss 1.6118186836286967 valid acc 15/16
Epoch 5 loss 1.1720250241650327 valid acc 15/16
Epoch 5 loss 1.0459072688182571 valid acc 15/16
Epoch 5 loss 1.0881033994819307 valid acc 15/16
Epoch 5 loss 0.44257082631277744 valid acc 14/16
Epoch 5 loss 0.7079143155788605 valid acc 15/16
Epoch 5 loss 0.8852291560694694 valid acc 15/16
Epoch 5 loss 0.8216213788185577 valid acc 15/16
Epoch 5 loss 0.8259662906686003 valid acc 16/16
Epoch 5 loss 1.370332405226126 valid acc 15/16
Epoch 5 loss 1.2196017655466944 valid acc 14/16
Epoch 5 loss 0.6330698138653144 valid acc 15/16
Epoch 5 loss 1.334059808060298 valid acc 15/16
Epoch 5 loss 0.6578043876139448 valid acc 14/16
Epoch 5 loss 0.8066221022157598 valid acc 15/16
Epoch 5 loss 0.9236338339877515 valid acc 13/16
Epoch 5 loss 1.4917576155086305 valid acc 14/16
Epoch 5 loss 0.7538553876189136 valid acc 15/16
Epoch 5 loss 1.0904827563155453 valid acc 16/16
Epoch 5 loss 1.4916110344539937 valid acc 14/16
Epoch 5 loss 1.469717673267056 valid acc 16/16
Epoch 5 loss 0.9584634261378343 valid acc 16/16
Epoch 5 loss 1.0049478103957628 valid acc 14/16
Epoch 5 loss 1.3569760401818385 valid acc 16/16
Epoch 5 loss 0.9308688962351963 valid acc 14/16
Epoch 5 loss 0.663496762150489 valid acc 14/16
Epoch 5 loss 1.5748745100311057 valid acc 15/16
Epoch 5 loss 0.42990214226693857 valid acc 16/16
Epoch 5 loss 0.7833198996630586 valid acc 16/16
Epoch 5 loss 1.278698247414566 valid acc 15/16
Epoch 5 loss 1.2019718856189097 valid acc 15/16
Epoch 5 loss 0.9699468454546702 valid acc 15/16
Epoch 5 loss 0.23560235670732024 valid acc 14/16
Epoch 5 loss 0.9692803860469964 valid acc 15/16
Epoch 5 loss 1.561499212437398 valid acc 15/16
Epoch 5 loss 0.9784282904182371 valid acc 15/16
Epoch 5 loss 0.9866116273578694 valid acc 15/16
Epoch 5 loss 0.6582831099119135 valid acc 15/16
Epoch 5 loss 0.5452779600183498 valid acc 15/16
Epoch 5 loss 1.0308597470324254 valid acc 16/16
Epoch 5 loss 1.2086017967586278 valid acc 15/16
Epoch 5 loss 1.210965279752549 valid acc 15/16
Epoch 5 loss 1.012671060792775 valid acc 16/16
Epoch 5 loss 0.7610467213332387 valid acc 15/16
Epoch 5 loss 1.3183213168652845 valid acc 16/16
Epoch 6 loss 0.020933939781447582 valid acc 16/16
Epoch 6 loss 1.177978146338766 valid acc 15/16
Epoch 6 loss 1.4907798944496422 valid acc 15/16
Epoch 6 loss 1.1535966508429656 valid acc 15/16
Epoch 6 loss 0.5249450164233749 valid acc 13/16
Epoch 6 loss 0.5715194131250232 valid acc 16/16
Epoch 6 loss 1.2581779277843985 valid acc 13/16
Epoch 6 loss 1.4539516276020366 valid acc 15/16
Epoch 6 loss 1.3431254120437925 valid acc 16/16
Epoch 6 loss 0.767277935726737 valid acc 14/16
Epoch 6 loss 0.5779060065290431 valid acc 15/16
Epoch 6 loss 1.823017408130199 valid acc 14/16
Epoch 6 loss 1.0590691684549505 valid acc 15/16
Epoch 6 loss 1.5636183325593667 valid acc 14/16
Epoch 6 loss 1.3851496329072805 valid acc 14/16
Epoch 6 loss 0.7918246229041581 valid acc 14/16
Epoch 6 loss 1.8466554061553908 valid acc 15/16
Epoch 6 loss 1.1652537508222194 valid acc 16/16
Epoch 6 loss 1.0652439349780707 valid acc 15/16
Epoch 6 loss 0.6345135787170305 valid acc 15/16
Epoch 6 loss 1.4136474943409159 valid acc 14/16
Epoch 6 loss 0.9187990798408718 valid acc 14/16
Epoch 6 loss 0.24894126380852827 valid acc 15/16
Epoch 6 loss 1.4181020105740318 valid acc 15/16
Epoch 6 loss 0.8350290667620455 valid acc 15/16
Epoch 6 loss 1.112429096220859 valid acc 15/16
Epoch 6 loss 0.4569246073832875 valid acc 15/16
Epoch 6 loss 1.0110104548065986 valid acc 13/16
Epoch 6 loss 1.310572553977638 valid acc 13/16
Epoch 6 loss 0.5189724171476799 valid acc 15/16
Epoch 6 loss 1.1123936669965822 valid acc 15/16
Epoch 6 loss 0.6153126666507508 valid acc 15/16
Epoch 6 loss 0.8217539746575598 valid acc 14/16
Epoch 6 loss 0.8244944041237368 valid acc 15/16
Epoch 6 loss 1.6284826173880087 valid acc 16/16
Epoch 6 loss 0.4782077900550268 valid acc 16/16
Epoch 6 loss 0.5665446792818009 valid acc 14/16
Epoch 6 loss 1.14856214849848 valid acc 13/16
Epoch 6 loss 1.1740140775353987 valid acc 16/16
Epoch 6 loss 1.0436161440828886 valid acc 15/16
Epoch 6 loss 0.7083607949521655 valid acc 16/16
Epoch 6 loss 1.1097649964194058 valid acc 15/16
Epoch 6 loss 0.8560143021749647 valid acc 16/16
Epoch 6 loss 0.674309386036191 valid acc 15/16
Epoch 6 loss 1.4312530602694031 valid acc 15/16
Epoch 6 loss 0.3715992684385564 valid acc 16/16
Epoch 6 loss 0.9583247129616546 valid acc 16/16
Epoch 6 loss 2.0703776669521092 valid acc 15/16
Epoch 6 loss 0.7600388264691256 valid acc 16/16
Epoch 6 loss 0.6996621934564966 valid acc 15/16
Epoch 6 loss 0.6015737426995069 valid acc 15/16
Epoch 6 loss 0.5753702331266327 valid acc 15/16
Epoch 6 loss 1.2102311182801504 valid acc 15/16
Epoch 6 loss 0.586023057625884 valid acc 15/16
Epoch 6 loss 1.172826949955414 valid acc 14/16
Epoch 6 loss 0.8028981107334339 valid acc 14/16
Epoch 6 loss 0.9073756065079995 valid acc 16/16
Epoch 6 loss 1.0444839664678298 valid acc 15/16
Epoch 6 loss 1.0748217485327671 valid acc 15/16
Epoch 6 loss 0.8694177825471329 valid acc 16/16
Epoch 6 loss 0.6690640041322493 valid acc 16/16
Epoch 6 loss 0.6223136057177915 valid acc 16/16
Epoch 6 loss 1.1887587297220354 valid acc 16/16
Epoch 7 loss 0.03890677397479855 valid acc 16/16
Epoch 7 loss 0.8757327692804494 valid acc 16/16
Epoch 7 loss 1.1119786831313414 valid acc 16/16
Epoch 7 loss 0.7495060909330363 valid acc 15/16
Epoch 7 loss 0.7730529664681647 valid acc 15/16
Epoch 7 loss 0.456513829488131 valid acc 16/16
Epoch 7 loss 0.8952340881149892 valid acc 15/16
Epoch 7 loss 0.9854905757152082 valid acc 14/16
Epoch 7 loss 1.1668736699733124 valid acc 16/16
Epoch 7 loss 0.7231470049460187 valid acc 13/16
Epoch 7 loss 0.854861626232608 valid acc 15/16
Epoch 7 loss 1.5217675661201344 valid acc 14/16
Epoch 7 loss 1.4394611032441096 valid acc 15/16
Epoch 7 loss 1.0764208887950097 valid acc 15/16
Epoch 7 loss 1.642865570749349 valid acc 15/16
Epoch 7 loss 0.7420135955123706 valid acc 13/16
Epoch 7 loss 2.1650718453294977 valid acc 14/16
Epoch 7 loss 0.8926153999422803 valid acc 15/16
Epoch 7 loss 0.9886560579262177 valid acc 14/16
Epoch 7 loss 0.5276512564123832 valid acc 15/16
Epoch 7 loss 1.008238923260338 valid acc 15/16
Epoch 7 loss 0.8640716783437652 valid acc 15/16
Epoch 7 loss 0.3481081452460277 valid acc 14/16
Epoch 7 loss 0.5744590380829147 valid acc 15/16
Epoch 7 loss 0.5043214662077847 valid acc 15/16
Epoch 7 loss 0.9067507415366359 valid acc 14/16
Epoch 7 loss 0.8119523042686926 valid acc 16/16
Epoch 7 loss 1.0070436059374797 valid acc 15/16
Epoch 7 loss 0.5355826564046298 valid acc 16/16
Epoch 7 loss 0.2638962111383977 valid acc 15/16
Epoch 7 loss 0.8089529380126743 valid acc 13/16
Epoch 7 loss 0.3085225189293283 valid acc 16/16
Epoch 7 loss 0.3878037821112354 valid acc 16/16
Epoch 7 loss 1.1527958219474843 valid acc 16/16
Epoch 7 loss 1.2024234778127962 valid acc 14/16
Epoch 7 loss 0.3852959410072524 valid acc 16/16
Epoch 7 loss 0.8422732261146323 valid acc 15/16
Epoch 7 loss 0.3556656119488334 valid acc 16/16
Epoch 7 loss 1.0360650238371498 valid acc 15/16
Epoch 7 loss 0.8149936792983089 valid acc 16/16
Epoch 7 loss 0.2536104552490954 valid acc 16/16
Epoch 7 loss 0.3940764491059136 valid acc 15/16
Epoch 7 loss 0.560152189575439 valid acc 16/16
Epoch 7 loss 0.3134104945000876 valid acc 15/16
Epoch 7 loss 1.5217802048583264 valid acc 16/16
Epoch 7 loss 0.23262624384840513 valid acc 16/16
Epoch 7 loss 0.530163283011893 valid acc 16/16
Epoch 7 loss 0.993239462206871 valid acc 16/16
Epoch 7 loss 0.5878315784247543 valid acc 15/16
Epoch 7 loss 0.9028330972324607 valid acc 16/16
Epoch 7 loss 0.33212266778583016 valid acc 16/16
Epoch 7 loss 0.7062717002757714 valid acc 16/16
Epoch 7 loss 1.2531925367018966 valid acc 16/16
Epoch 7 loss 0.7837307616702355 valid acc 16/16
Epoch 7 loss 0.9899613091170429 valid acc 16/16
Epoch 7 loss 0.6215747717881002 valid acc 14/16
Epoch 7 loss 0.4931181594482658 valid acc 16/16
Epoch 7 loss 0.8136615134787591 valid acc 15/16
Epoch 7 loss 1.3261555009620758 valid acc 15/16
Epoch 7 loss 0.5781232041839992 valid acc 15/16
Epoch 7 loss 0.6854506453720841 valid acc 14/16
Epoch 7 loss 0.3810013195812618 valid acc 15/16
Epoch 7 loss 0.708864435231424 valid acc 16/16