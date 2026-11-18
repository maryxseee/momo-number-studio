/* Image preparation is JavaScript; neural inference is ONNX Runtime Web. */
export function validateNumber(text){
  if(!/^[0-9]{1,4}$/.test(text))return {valid:false,message:'Не вдалося скласти число з 1–4 цифр.'};
  if(text.length>1&&text[0]==='0')return {valid:false,message:'Напишіть число без провідних нулів.'};
  const number=Number(text);
  if(number<1||number>1000)return {valid:false,message:'Результат поза діапазоном 1–1000.'};
  return {valid:true,number,message:null};
}

export function segmentNumber(pixels,width,height){
  if(pixels.length!==width*height||width<28||width>1120||height<28||height>560)throw Error('Некоректний розмір полотна.');
  const mask=new Uint8Array(pixels.length),queue=new Int32Array(pixels.length),clean=new Uint8Array(pixels.length);
  let total=0;
  for(let i=0;i<pixels.length;i++){if(!Number.isFinite(pixels[i])||pixels[i]<0||pixels[i]>255)throw Error('Некоректні пікселі.');if(pixels[i]>40){mask[i]=1;total++;}}
  const minimum=Math.max(6,Math.floor(total*.002));
  for(let i=0;i<mask.length;i++)if(mask[i]===1){
    let head=0,tail=1;queue[0]=i;mask[i]=2;
    while(head<tail){const p=queue[head++],x=p%width,y=Math.floor(p/width);
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,yy=y+dy;if(xx<0||xx>=width||yy<0||yy>=height)continue;const k=yy*width+xx;if(mask[k]===1){mask[k]=2;queue[tail++]=k;}}
    }
    if(tail>=minimum)for(let k=0;k<tail;k++)clean[queue[k]]=pixels[queue[k]];
  }
  const columns=[];let yMin=height,yMax=-1,ink=0;
  for(let x=0;x<width;x++){let active=false;for(let y=0;y<height;y++)if(clean[y*width+x]){active=true;ink++;yMin=Math.min(yMin,y);yMax=Math.max(yMax,y);}if(active)columns.push(x);}
  if(ink<10)throw Error('Спочатку намалюйте число.');
  const gap=Math.max(4,Math.round((yMax-yMin+1)*.035)),groups=[];
  let start=columns[0],previous=start;
  for(const x of columns.slice(1)){if(x-previous>gap+1){groups.push([start,previous]);start=x;}previous=x;}groups.push([start,previous]);
  if(groups.length>4)throw Error('Знайдено більше чотирьох цифр. Напишіть одне число.');
  return groups.map(([left,right])=>{let top=height,bottom=-1;for(let y=0;y<height;y++)for(let x=left;x<=right;x++)if(clean[y*width+x]){top=Math.min(top,y);bottom=Math.max(bottom,y);}return {bbox:[left,top,right-left+1,bottom-top+1],pixels:clean,width,height};});
}

export function prepareGlyph(part){
  const [left,top,w,h]=part.bbox,source=document.createElement('canvas');source.width=w;source.height=h;
  const sc=source.getContext('2d'),image=sc.createImageData(w,h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const value=part.pixels[(top+y)*part.width+left+x],i=(y*w+x)*4;image.data[i]=image.data[i+1]=image.data[i+2]=value;image.data[i+3]=255;}
  sc.putImageData(image,0,0);
  const out=document.createElement('canvas');out.width=out.height=28;
  const ctx=out.getContext('2d');ctx.fillStyle='#000';ctx.fillRect(0,0,28,28);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';
  const scale=20/Math.max(w,h),rw=Math.max(1,Math.round(w*scale)),rh=Math.max(1,Math.round(h*scale));
  ctx.drawImage(source,Math.floor((28-rw)/2),Math.floor((28-rh)/2),rw,rh);
  const values=ctx.getImageData(0,0,28,28).data,raw=new Float32Array(784);
  let mass=0,cx=0,cy=0;
  for(let y=0;y<28;y++)for(let x=0;x<28;x++){const v=values[(y*28+x)*4]/255;raw[y*28+x]=v;mass+=v;cx+=x*v;cy+=y*v;}
  if(mass===0)throw Error('Штрих надто малий. Спробуйте намалювати більшу цифру.');
  const dx=13.5-cx/mass,dy=13.5-cy/mass,centered=new Float32Array(784);
  const at=(x,y)=>x<0||x>27||y<0||y>27?0:raw[y*28+x];
  for(let y=0;y<28;y++)for(let x=0;x<28;x++){const sx=x-dx,sy=y-dy,x0=Math.floor(sx),y0=Math.floor(sy),fx=sx-x0,fy=sy-y0;centered[y*28+x]=at(x0,y0)*(1-fx)*(1-fy)+at(x0+1,y0)*fx*(1-fy)+at(x0,y0+1)*(1-fx)*fy+at(x0+1,y0+1)*fx*fy;}
  return centered;
}

let sessionPromise;
export function loadModel(){
  if(!sessionPromise){
    if(!globalThis.ort)throw Error('Не вдалося завантажити бібліотеку моделі. Оновіть сторінку.');
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
    sessionPromise=ort.InferenceSession.create(new URL('./assets/digit-cnn.onnx',import.meta.url).href,{executionProviders:['wasm']}).catch(error=>{sessionPromise=null;throw error;});
  }
  return sessionPromise;
}

export async function predict(pixels,width,height,mode='number'){
  let parts;
  if(mode==='digit'){
    let left=width,top=height,right=-1,bottom=-1,ink=0;
    for(let y=0;y<height;y++)for(let x=0;x<width;x++)if(pixels[y*width+x]>30){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);ink++;}
    if(ink<10)throw Error('Спочатку намалюйте цифру.');
    parts=[{bbox:[left,top,right-left+1,bottom-top+1],pixels,width,height}];
  }else parts=segmentNumber(pixels,width,height);
  const inputs=parts.map(prepareGlyph),batch=new Float32Array(parts.length*784);inputs.forEach((image,i)=>batch.set(image,i*784));
  const session=await loadModel(),output=await session.run({pixels:new ort.Tensor('float32',batch,[parts.length,1,28,28])});
  const digits=parts.map((part,i)=>({bbox:part.bbox,preview:inputs[i],top3:Array.from({length:10},(_,digit)=>({digit,score:output.probabilities.data[i*10+digit]})).sort((a,b)=>b.score-a.score).slice(0,3)}));
  const text=digits.map(d=>d.top3[0].digit).join('');
  return {text,digits,...(mode==='number'?validateNumber(text):{valid:true,number:Number(text),message:null})};
}
