# ベースイメージにpython3を使用
FROM python:3

# 深層学習で画像認識を行うためのパッケージをインストール
RUN apt-get update && \
	apt-get install -y libgl1-mesa-glx && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	pip install --upgrade pip && \
	pip install numpy pandas matplotlib seaborn scipy opencv-python scikit-learn torch torchvision tqdm jupyterlab && \
	rm -rf ~/.cache/pip && \
	mkdir work

WORKDIR /work

# デフォルトはjupyterlabを起動するように指定
CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]