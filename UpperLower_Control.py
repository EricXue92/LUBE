import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import backend as K

class UpperLower_Control(keras.Model):

    def init_arguments(self, method = 'No_PCGrad', coverage_rate = 0.96, lamda = 0.1):

        self.method = method  
        self.coverage_rate = coverage_rate
        self.lamda = lamda 

    # Define selective loss function 
    def selective_up(self, y_true, y_pred):
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:,1] - y_true[:,0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:,0] - y_pred[:,0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        ### 
        width_up_loss = K.square(y_pred[:,1]-y_true[:,0])
        # Only compute selected ones 
        width_up_loss = tf.math.multiply(indicator, width_up_loss)
        width_up_loss = (K.sum(width_up_loss) + K.epsilon())/(K.sum(indicator) + K.epsilon())
        return width_up_loss 
 
    def selective_low(self, y_true, y_pred):
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:,1] - y_true[:,0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:,0] - y_pred[:,0]), 0.5), K.floatx())
        indicator = tf.multiply(ub_ind, lb_ind)
        width_low_loss = K.square(y_true[:,0] - y_pred[:,0])
        # Only compute selected ones 
        width_low_loss = tf.math.multiply(indicator, width_low_loss)
        width_low_loss = (K.sum(width_low_loss) + K.epsilon())/(K.sum(indicator) + K.epsilon())
        return width_low_loss

    # Upper penalty function (upper_bound >= y_true)
    def up_penalty(self, y_true,  y_pred):
        up_penalty_loss = K.sum( K.maximum( [0.0], (y_true[:, 1] - y_pred[:, 1]))) 
        return up_penalty_loss

    # Lower penalty function (lower_bound =< y_true)
    def low_penalty(self, y_true,  y_pred):
        low_penalty_loss = K.sum(K.maximum( [0.0], y_pred[:,0] - y_true[:, 1] ))
        return low_penalty_loss

    # Coverage penalty function
    def coverage_penalty(self, y_true, y_pred):
        ub_ind = K.cast(K.greater(K.sigmoid(y_pred[:,1] - y_true[:,0]), 0.5), K.floatx())
        lb_ind = K.cast(K.greater(K.sigmoid(y_true[:,0] - y_pred[:,0]), 0.5), K.floatx())
        coverage_value = K.mean(tf.multiply(ub_ind, lb_ind))
        #return self.lamda * (K.exp(K.maximum(0.0, self.coverage_rate - coverage))-1)
        return  self.lamda * (K.maximum(0.0,  self.coverage_rate - coverage_value))

    # Calculate the coverage 
    def coverage(self, y_true, y_pred):
        coverage_value = K.cast( K.mean((y_pred[:,1] >= y_true[:,0]) & (y_true[:,0]>=y_pred[:,0]) ), K.floatx())
        return coverage_value

    # Calculate Mean predicition interval width(mpiw)
    def mpiw(self, y_true, y_pred):
        res = K.cast( K.mean(K.square( y_pred[:,0] - y_pred[:,1])), K.floatx())
        return res 

#   @article{yu2020gradient,
#   title={Gradient surgery for multi-task learning},
#   author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
#   journal={arXiv preprint arXiv:2001.06782},
#   year={2020}}

    def compute_gradients_by_PCG(self, loss):
        assert type(loss) is list
        loss = tf.stack(loss)
        tf.random.shuffle(loss)
        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1,])
                            for grad in tf.gradients(x, self.trainable_variables)
                            if grad is not None], axis=0), loss)
        num_tasks = loss.shape[0]
        # Compute gradient projections.
        def proj_grad(grad_task):
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(grad_task*grads_task[k])
                proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
            return grad_task
        proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)
         # Unpack flattened projected gradients back to their original shapes.
        proj_grads = []
        for j in range(num_tasks):
            start_idx = 0
            for idx, var in enumerate(self.trainable_variables):
                grad_shape = var.get_shape()
                flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                proj_grad = tf.reshape(proj_grad, grad_shape)
                if len(proj_grads) < len(self.trainable_variables):
                    proj_grads.append(proj_grad)
                else:
                    proj_grads[idx] += proj_grad               
                start_idx += flatten_dim

        grads_and_vars = zip(proj_grads, self.trainable_variables)
        return proj_grads


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        X, y = data 
        y_pred = self(X, training=True)  # Forward pass

        # Compute the loss value (the loss function is configured in `compile()`)
        width_up_loss = self.selective_up(y, y_pred)
        width_low_loss = self.selective_low(y, y_pred)
        up_penalty_loss = self.up_penalty(y, y_pred)
        low_penalty_loss = self.low_penalty(y, y_pred)
        coverage_penalty = self.coverage_penalty(y, y_pred)
        
        # Calculate the metrics 
        coverage_value = self.coverage(y, y_pred)
        mpiw_value = self.mpiw(y, y_pred)

        trainable_vars = self.trainable_variables

        if self.method == 'No_PCGrad':
            loss = 1/4*K.mean(width_up_loss)+1/4*K.mean(width_low_loss)+1/4*coverage_penalty*K.mean(up_penalty_loss)+1/4*coverage_penalty*K.mean(low_penalty_loss)
            gradients = tf.gradients(loss, trainable_vars)

        elif self.method == 'PCGrad':
            loss = [K.mean(width_up_loss) + K.mean(width_low_loss), ( K.mean(low_penalty_loss)+K.mean(up_penalty_loss) )*K.mean(coverage_penalty) +K.epsilon()]
            #loss = [K.mean(width_up_loss)+K.mean(low_penalty_loss)*(K.mean(coverage_penalty) + K.epsilon()), K.mean(width_low_loss)+K.mean(up_penalty_loss)*(K.mean(coverage_penalty) + K.epsilon())]
            gradients = self.compute_gradients_by_PCG(loss)

        # To debug 
        # tf.print(' Width_Loss:', width_up_loss+width_low_loss, 
        #     ' Penalty_Loss:', up_penalty_loss+low_penalty_loss, 
        #     ' Coverage_penalty_Loss:', coverage_penalty,
        #     'Sum Loss:', width_up_loss+width_low_loss+coverage_penalty*up_penalty_loss+coverage_penalty*low_penalty_loss)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}