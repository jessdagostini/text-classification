# Text Classification Model com TensorFlow

Trabalho final da disciplina de Inteligência Artificial - Ano 2018

---

Como executar:

- No Windows
    
    1) Faça a instalação do Anaconda https://www.anaconda.com/download/. Basta seguir os passos do instalador.

    2) Depois de instalado, abra o terminal do Windows e rode o seguinte comando:
        
            conda create -n tensorflow pip python=3.5

        para criar um ambiente no conda chamado TensorFlow, que será onde a biblioteca será instalada.
    
    3) Em seguida, ative o ambiente com

            activate tensorflow

    4) Por fim, instale a biblioteca TensorFlow:

            pip install --ignore-installed --upgrade tensorflow
    5) Faça o teste da instalação com:

    ```
    python

    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> print(sess.run(hello))
    ```

    Você deve ver a mensagem

        Hello, TensorFlow!
    
    6) Agora precisamos abrir o spyder. Para isso, abra o Anaconda Navigator, vá em "Environments" e escolha a opção "Tensorflow".

    7) Após a troca, volte para a Home e procure pelo Spyder. No seu primeiro acesso, provavelmente será necessário instalar ele.

Com o Spyder aberto, procure pela aba "Explorador de Arquivos" e navegue até a pasta que contém os arquivos `imdb-classification.py` e os datasets.

> REFERÊNCIAS

- Dataset: 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

- Text Classification Model: https://medium.com/@dehhmesquita/classificando-textos-com-redes-neurais-e-tensorflow-5063784a1b31