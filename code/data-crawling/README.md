# 根据关键字爬取数据集

### 1. 环境配置
这个使用的是火狐浏览器，运行会自己打开火狐浏览器，因此需要驱动
```python
# 启动Firefox浏览器  
driver = webdriver.Firefox()
```
[geckodriver驱动下载地址](https://github.com/mozilla/geckodriver/releases)    

根据系统下载好，放在火狐浏览器安装跟目录，如  
*C:\Program Files\Mozilla Firefox\geckodriver.exe*  
**并且需要配置环境变量到path**


### 2. 代码关键配置项
```python
#输出目录
OUTPUT_DIR = '../../data/raw/'
#关键字数组：将在输出目录内创建以以下关键字们命名的txt文件
SEARCH_KEY_WORDS = ['路飞','娜美', '索隆', '乔巴', '罗宾']
#页数
PAGE_NUM = 12
```

### 3. 检索路径
检索的是google图片搜索
```python
def getSearchUrl(keyWord):
    if(isEn(keyWord)):
        return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&source=lnms&tbm=isch'
    else:
        return 'https://www.google.com.hk/search?q=' + keyWord + '&safe=strict&hl=zh-CN&source=lnms&tbm=isch'
```
