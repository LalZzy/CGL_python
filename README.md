# CGL_python
concept graph learning

This is an python version inplementation of [CGL(concept graph learning)](https://github.com/quark0/CGL).
> Hanxiao Liu, Wanli Ma, Yiming Yang, and Jaime Carbonell. "Learning Concept Graphs from Online Educational Data." In Journal of Artificial Intelligence Research 55 (2016): 1059-1090. 
[PDF](http://www.cs.cmu.edu/~hanxiaol/publications/liu-jair16.pdf)


download it, cd into the directory and run
```shell
pip3 install -r requirement.txt
```

* STEP 1:  cd `'./data'`
* STEP 2:  make sure everybook's word_info file exists: `'finals/[bookname]_final.csv'`
and all concepts' json file exists:`'./all_concepts.json'`
* STEP 3: according to priori knowledge, give the prerequiste relations between and chapters:`'chapter_prerequisite.csv'`
* STEP 4: run `python generate_link.py` 
* STEP 6: cd `'../'` （return to parent directory）
* STEP 7: run `python main.py`
