# Model-agnostic Meta Learning Code - Classification

## MAML (Finn et al., 2017)
- Model-agnostic Meta Learning: 모델에 관계없는(-agnostic) 메타러닝 방법. 새로운 테스크에 잘 대응할 수 있는 (fast adaptation) 모델 파라미터를 학습시키는 것에 초점을 맞춤.

- 우리가 익숙한 방법과 다른점
	- supervised learning과 달리 meta learning에서는 하나의 클래스당 적은 데이터 밖에 없음 
-> 코드 작성시 dataloader 부분에서 작성 유의, meta-learning에 적합한 문제를 설정해야함.
	- 모델의 파라미터 (theta)를 업데이트하기 위해서는 T_i로 학습시킨 theta_i의 합을 통함.  기존의 multi-task learning의 경우 여러개의 task에 대해서 당장 업데이트 로스를 최소화할 수 있도록 학습이 되었다면, meta-learning의 경우 주어진 training task를 수행한 뒤에 해당 loss의 합에 대한 업데이트 로스를 최소화한다는 점
-> 하나의 테스크 내부에서는 우리가 익숙한 방식(하나의 테스크를 훈련했던 방식)대로 훈련을 하면되지만, 코드 작성시 전체 테스크의 loss를 통합해서 optimizer에 반영하는 부분이 추가되어야 함.

- MAML 모델에서 한 epoch은 meta-leaner와 sub-learners로 구성됨
![maml diagram](https://github.com/Higgsboson-X/maml-cnn-text-classifier/blob/master/images/maml_fig.png "MAML Diagram")
	**1. Meta Learning Stage**
	- Meta parameter를 sub-learners를 통해 얻어내는 과정
	- sub-learner는 모두 동일한 모델 파라미터에서 시작
	- Sub-learner는 각각 주어진 하나의 테스크를 적은 update 횟수(아래 코드에서는 5번)을 통해 task-specific parameters and loss (theta_i, loss_i)를 구한다
	- loss_0, …, loss_i의 합을 theta optimizer에 반영
	- code example: [reference](https://github.com/dragen1860/MAML-Pytorch/blob/98a00d41724c133bd29619a2fb2cc46dd128a368/meta.py#L65-L147)
  
  ```
      def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        
        for i in range(task_num):
  
              for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                # 모델 파라미터 (fast_weights)에 대해 loss 계산
                grad = torch.autograd.grad(loss, fast_weights) 
                # 3. theta_pi = theta_pi - train_lr * grad
                # 계산된 loss를 통해 파라미터 업데이트 
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        # 마지막 query set을 통해 구해진 loss는 해당 서브 테스크의 loss를 계산 하는데 사용되며,
        # 이렇게 구해진 서브 테스크들의 loss의 평균이 meta model parameters에 반영됨.
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad() # gradient를 0으로 초기화
        loss_q.backward() # loss_q 값으로 모델 파라이터들의 loss 구하기
        # print('meta update')
        self.meta_optim.step() # 계산된 loss로 meta model parmeters로 업데이트
  ```
  
 
	**2. Fine-tuning Stage**
	- 모델을 meta-learning stage에서 얻은 파라미터로 초기값 설정 -> Quick adaption 효과를 내기 위함
	- 주어진 적은 양의 데이터만으로도 유의미한 성능을 얻도록 함
  - code example: [reference](https://github.com/dragen1860/MAML-Pytorch/blob/98a00d41724c133bd29619a2fb2cc46dd128a368/meta.py#L150-L218)
	
  ```
      def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
  ```
  
  **3. Meta Learning Stage, Fine-tuning Stage에서 사용하는 데이터는 동일**

- MAML 알고리즘
![maml algorithm](https://github.com/Higgsboson-X/maml-cnn-text-classifier/blob/master/images/maml_alg.png "MAML Algorithm")


## Reference
- MAML 설명 + Tensorflow code: [GitHub - Higgsboson-X/maml-cnn-text-classifier: Model-agnostic meta-learning framework adapted to CNN text classifier](https://github.com/Higgsboson-X/maml-cnn-text-classifier)
- Torch Code (referred in this post): [GitHub - dragen1860/MAML-Pytorch: Elegant PyTorch implementation of paper Model-Agnostic Meta-Learning (MAML)](https://github.com/dragen1860/MAML-Pytorch)
- 한국어 설명 (Regression with Time Series Data): [Model Agnostic Meta-Learning (Pytorch) | Deep Chioni Blog](https://chioni.github.io/posts/mamlp/)

