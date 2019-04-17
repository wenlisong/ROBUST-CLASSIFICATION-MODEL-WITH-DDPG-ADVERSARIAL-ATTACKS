####noise + image action/255
ssh wlsong@gpu18.cse.cuhk.edu.hk
screen -r
CUDA_VISIBLE_DEVICES=1 python ddpg4.py

####image
ssh wlsong@gpu19.cse.cuhk.edu.hk
screen -r
CUDA_VISIBLE_DEVICES=0 python ddpg3.py
rm -rfv ./output-example3/*

wlsong@gpu19.cse.cuhk.edu.hk:/research/ksleung5/wlsong/ROBUST-CLASSIFICATION-MODEL-WITH-DDPG-ADVERSARIAL-ATTACKS/output-example
CUDA_VISIBLE_DEVICES=0 python 

git pull origin master
git fetch
git merge origin/master
git add .
git commit -m '
git push -u origin master


mkdir $(printf "%05i " $(seq 0 119))

##### these two images should be remove
/00012/1e19e7fa7da1641e786b69dc8eed9daa.jpg
/00092/bdc7be7063d7e99953bbaee2cc99888c.jpg
