/* Genuine last-layer training through TensorFlow.js; feature CNN stays frozen. */
export class PersonalHead{
  constructor(tf,initial,replay,options={}){
    this.tf=tf;this.initial=initial;this.replay=replay;
    this.options={epochs:12,learningRate:.001,guardTolerance:.03,...options};
    this.model=tf.sequential();
    this.model.add(tf.layers.dense({inputShape:[initial.featureSize],units:10,activation:'softmax',kernelInitializer:'zeros',biasInitializer:'zeros'}));
    this.restore([initial.kernel,initial.bias]);
    this.baseGuardAccuracy=this.accuracy(replay.guard);
  }
  restore(weights){
    const tf=this.tf;
    tf.tidy(()=>this.model.setWeights([tf.tensor2d(weights[0],[this.initial.featureSize,10]),tf.tensor1d(weights[1])]));
  }
  weights(){return this.tf.tidy(()=>this.model.getWeights().map(t=>Array.from(t.dataSync())));}
  predict(features){
    return this.tf.tidy(()=>Float32Array.from(this.model.predict(this.tf.tensor2d(features,[features.length,this.initial.featureSize])).dataSync()));
  }
  accuracy(dataset){
    const scores=this.predict(dataset.features);let correct=0;
    dataset.labels.forEach((label,i)=>{let best=0;for(let d=1;d<10;d++)if(scores[i*10+d]>scores[i*10+best])best=d;if(best===label)correct++;});
    return correct/dataset.labels.length;
  }
  async train(examples){
    if(!examples.length||examples.some(e=>!Number.isInteger(e.label)||e.label<0||e.label>9||e.features.length!==this.initial.featureSize||e.features.some(v=>!Number.isFinite(v))))throw Error('Некоректний навчальний приклад.');
    const tf=this.tf,features=this.replay.train.features.map(row=>row.slice()),labels=this.replay.train.labels.slice();
    // Replay original train-only anchors and all recent corrections; no test data.
    for(let repeat=0;repeat<8;repeat++)for(const example of examples){features.push(example.features);labels.push(example.label);}
    const x=tf.tensor2d(features,[features.length,this.initial.featureSize]);
    const y=tf.tidy(()=>tf.oneHot(tf.tensor1d(labels,'int32'),10));
    const snapshot=this.weights(),guardBefore=this.accuracy(this.replay.guard);
    this.optimizer?.dispose();this.optimizer=tf.train.adam(this.options.learningRate);
    this.model.compile({optimizer:this.optimizer,loss:'categoricalCrossentropy'});
    const loss=()=>tf.tidy(()=>this.model.evaluate(x,y).dataSync()[0]);
    const lossBefore=loss();
    try{
      await this.model.fit(x,y,{epochs:this.options.epochs,batchSize:64,shuffle:true,verbose:0,callbacks:{onEpochEnd:async()=>{await tf.nextFrame();}}});
      const lossAfter=loss(),guardAfter=this.accuracy(this.replay.guard);
      const accepted=Number.isFinite(lossAfter)&&guardAfter+1e-7>=this.baseGuardAccuracy-this.options.guardTolerance;
      if(!accepted)this.restore(snapshot);
      return {accepted,lossBefore,lossAfter,guardBefore,guardAfter,baseGuardAccuracy:this.baseGuardAccuracy,epochs:this.options.epochs};
    }catch(error){this.restore(snapshot);throw error;}
    finally{x.dispose();y.dispose();}
  }
  dispose(){this.optimizer?.dispose();this.model.dispose();}
}
