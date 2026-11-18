import {PersonalHead} from './personal-head.mjs';
const DB='momo-number-studio-personal-v1',KEY='profile';
let profile=null,head=null,tfPromise=null,assetsPromise=null,persisted=true;

function storage(mode,value){
  return new Promise((resolve,reject)=>{
    if(!globalThis.indexedDB){reject(Error('Локальне сховище браузера недоступне.'));return;}
    const open=indexedDB.open(DB,1);
    open.onupgradeneeded=()=>open.result.createObjectStore('profile');
    open.onerror=()=>reject(open.error);
    open.onsuccess=()=>{
      const db=open.result,tx=db.transaction('profile',mode==='read'?'readonly':'readwrite'),store=tx.objectStore('profile');
      const request=mode==='read'?store.get(KEY):mode==='delete'?store.delete(KEY):store.put(value,KEY);
      let result;request.onsuccess=()=>result=request.result;
      tx.oncomplete=()=>{db.close();resolve(result);};tx.onerror=()=>{db.close();reject(tx.error);};
      tx.onabort=()=>{db.close();reject(tx.error||Error('Збереження перервано.'));};
    };
  });
}
async function tensorflow(){
  if(!tfPromise)tfPromise=(async()=>{
    if(!globalThis.tf)await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=new URL('./vendor/tf.min.js',import.meta.url).href;script.onload=resolve;script.onerror=()=>{script.remove();reject(Error('Не вдалося завантажити бібліотеку навчання.'));};document.head.append(script);});
    await tf.setBackend('cpu');await tf.ready();return tf;
  })().catch(error=>{tfPromise=null;throw error;});
  return tfPromise;
}
async function assets(){
  if(!assetsPromise)assetsPromise=Promise.all(['head-initial.json','head-replay.json'].map(async name=>{const response=await fetch(new URL('./assets/'+name,import.meta.url));if(!response.ok)throw Error('Не вдалося завантажити матеріали для навчання.');return response.json();})).catch(error=>{assetsPromise=null;throw error;});
  return assetsPromise;
}
async function ensureHead(){
  if(head)return head;
  const [initial,replay]=await assets(),library=await tensorflow();
  if(initial.modelTag!==replay.modelTag)throw Error('Версії матеріалів для навчання не збігаються. Оновіть сторінку.');
  head=new PersonalHead(library,initial,replay);
  if(profile?.modelTag===initial.modelTag&&profile.weights)head.restore(profile.weights);
  return head;
}
export function personalStatus(){return {active:Boolean(profile?.active),updates:profile?.updates||0,examples:profile?.examples?.length||0,persisted};}
export async function initializeProfile(){
  try{
    profile=await storage('read')||null;
    if(profile){
      const [initial]=await assets();
      if(profile.schema!==1||profile.modelTag!==initial.modelTag){profile=null;return {...personalStatus(),warning:'Базову модель оновлено. Почніть персональне навчання для нової версії.'};}
      if(profile.active)await ensureHead();
    }
    return personalStatus();
  }catch(error){profile=null;head?.dispose();head=null;persisted=false;return {...personalStatus(),warning:'Браузер не дозволив відновити локальне навчання. Працює базова модель.'};}
}
export function personalScores(features){return profile?.active&&head?head.predict(features):null;}
export async function learnCorrection(features,text){
  if(!/^[0-9]+$/.test(text))throw Error('Вкажіть правильний запис цифрами 0–9.');
  if(features.length!==text.length)throw Error(`Виділено фрагментів: ${features.length}, а у виправленні цифр: ${text.length}. Перемалюйте з проміжками між цифрами: без правильного поділу навчання не виконується.`);
  const classifier=await ensureHead(),[initial]=await assets();
  const added=Array.from(text,(digit,i)=>({features:Array.from(features[i]),label:Number(digit)}));
  const examples=[...(profile?.examples||[]),...added].slice(-120);
  const diagnostics=await classifier.train(examples);
  profile={schema:1,modelTag:initial.modelTag,active:Boolean(profile?.active)||diagnostics.accepted,
    updates:(profile?.updates||0)+(diagnostics.accepted?1:0),examples,weights:classifier.weights(),
    lastTraining:diagnostics,updatedAt:new Date().toISOString()};
  try{await storage('write',profile);persisted=true;}catch{persisted=false;}
  return {...diagnostics,...personalStatus()};
}
export async function resetPersonalization(){
  await storage('delete');head?.dispose();head=null;profile=null;persisted=true;return personalStatus();
}
