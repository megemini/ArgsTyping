# ArgsTyping

通过 docstring 中的 `Args/Parameters` 和 `Returns/Yields` 进行类型标注。

可以直接覆盖原文件

``` shell
> python args_typing.py -i /home/shun/Documents/Projects/paddle/megemini/Paddle/python/paddle/tensor/math.py
```

或者手动设置输出文件

``` shell
> python args_typing.py -i /home/shun/Documents/Projects/paddle/megemini/Paddle/python/paddle/tensor/math.py -o ./tmp.py
```

## 注意

- 此脚本只做简单的类型标注，标注的正确与否仍需通过 mypy 等工具进行确认。
