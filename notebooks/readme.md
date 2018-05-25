# Howto
## Add .svg to Notebook
Upload the .svg file to github. Add the following code snippet adjusted to your file location into a markdown cell and run it.
```
<img src="https://raw.githubusercontent.com/arturmiller/MachineLearning/master/notebooks/images/classification_as_inversion.svg?sanitize=true" style="width: 100%;"/> 
```

## Create HTML
If you want to render your notebook without css-style run following command. 
```
jupyter nbconvert --to html --template basic fizz_buzz.ipynb
```
## Add style
Add the following style in front of your html content.
```
<style type="text/css">
.highlight{background: #f8f8f8; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .1em;padding:0em .5em;border-radius: 4px;}
.k{color: #338822; font-weight: bold;}
.kn{color: #338822; font-weight: bold;}
.mi{color: #000000;}
.o{color: #000000;}
.ow{color: #BA22FF;  font-weight: bold;}
.nb{color: #338822;}
.n{color: #000000;}
.s{color: #cc2222;}
.se{color: #cc2222; font-weight: bold;}
.si{color: #C06688; font-weight: bold;}
.nn{color: #4D00FF; font-weight: bold;}
</style>
```
