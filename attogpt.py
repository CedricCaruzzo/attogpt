import os,math as M,random as R;R.seed(42)
l='input.txt'
if not os.path.exists(l):import urllib.request as U;U.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',l)
D=[x.strip()for x in open(l)if x.strip()];R.shuffle(D);T=sorted(set(''.join(D)));V,B=len(T)+1,len(T)
class E:
 def __init__(s,d,c=(),g=()):s.data,s.grad,s._c,s._g=d,0,c,g
 __add__=lambda s,o:E(s.data+(o.data if hasattr(o,'data')else o),(s,o if hasattr(o,'data')else E(o)),(1,1))
 __mul__=lambda s,o:E(s.data*(q:=o.data if hasattr(o,'data')else o),(s,o if hasattr(o,'data')else E(o)),(q,s.data))
 __pow__=lambda s,p:E(s.data**p,(s,),(p*s.data**(p-1),));__truediv__=lambda s,o:s*o**-1;__neg__=lambda s:s*-1;log=lambda s:E(M.log(s.data),(s,),(1/s.data,));exp=lambda s:E(M.exp(s.data),(s,),(M.exp(s.data),));relu=lambda s:E(max(0,s.data),(s,),(1.*(s.data>0),))
 __radd__=__add__;__rmul__=__mul__;__sub__=lambda s,o:s+-o;__rsub__=lambda s,o:o+-s;__rtruediv__=lambda s,o:o*s**-1
 def backward(s):t=[];v=set();f=lambda x:(v.add(x),[f(c)for c in x._c],t.append(x))if x not in v else 0;f(s);s.grad=1;[setattr(c,'grad',c.grad+g*n.grad)for n in t[::-1]for c,g in zip(n._c,n._g)]
n,h,Y=16,4,16;H=n//h;G=lambda r,c,s=.08:[[E(R.gauss(0,s))for _ in range(c)]for _ in range(r)];C={'wte':G(V,n),'wpe':G(Y,n),'lm_head':G(V,n),'layer0.attn_wq':G(n,n),'layer0.attn_wk':G(n,n),'layer0.attn_wv':G(n,n),'layer0.attn_wo':G(n,n),'layer0.mlp_fc1':G(4*n,n),'layer0.mlp_fc2':G(n,4*n)};W=[p for m in C.values()for r in m for p in r]
I=lambda x,w:[sum(a*b for a,b in zip(r,x))for r in w];Z=lambda l:(lambda m,e,s:[x/s for x in e])(max(a.data for a in l),[(a-max(a.data for a in l)).exp()for a in l],sum([(a-max(a.data for a in l)).exp()for a in l]))
P=lambda x:[v*(sum(u*u for u in x)/len(x)+1e-5)**-.5 for v in x];A=lambda a,b:[x+y for x,y in zip(a,b)]
def Q(t,p,K,J):
 x=P(A(C['wte'][t],C['wpe'][p]));x_r=x;x=P(x);q,k,v=I(x,C['layer0.attn_wq']),I(x,C['layer0.attn_wk']),I(x,C['layer0.attn_wv']);K.append(k);J.append(v);r=[]
 for i in range(h):e=i*H;w=q[e:e+H];ks=[a[e:e+H]for a in K];vs=[a[e:e+H]for a in J];sc=[sum(w[j]*ks[u][j]for j in range(H))/H**.5 for u in range(len(ks))];sp=Z(sc);r.extend([sum(sp[u]*vs[u][j]for u in range(len(vs)))for j in range(H)])
 x=A(I(r,C['layer0.attn_wo']),x_r);x_r=x;x=P(x);x=I(x,C['layer0.mlp_fc1']);x=[a.relu()for a in x];x=I(x,C['layer0.mlp_fc2']);x=A(x,x_r);return I(x,C['lm_head'])
m,v,f=[0.]*len(W),[0.]*len(W),1000;lr,b1,b2,ep=.01,.85,.99,1e-8
for s in range(f):
 z=D[s%len(D)];g=[B]+[T.index(c)for c in z]+[B];o=min(Y,len(g)-1);K,J=[],[];L=sum([list(Z(Q(g[i],i,K,J)))[g[i+1]].log()*-1 for i in range(o)])/o;L.backward();lr_t=lr*(1-s/f)
 for i,w in enumerate(W):m[i]=b1*m[i]+(1-b1)*w.grad;v[i]=b2*v[i]+(1-b2)*w.grad**2;m_h=m[i]/(1-b1**(s+1));v_h=v[i]/(1-b2**(s+1));w.data-=lr_t*m_h/(v_h**.5+ep);w.grad=0
 print(f"step {s+1:4d} / {f:4d} | loss {L.data:.4f}")
print('\n--- inference ---')
for j in range(20):
 K,J=[],[];t=B;r=[]
 for p in range(Y):
  t=R.choices(range(V),[a.data for a in Z([a/.5 for a in Q(t,p,K,J)])])[0]
  if t==B:break
  r.append(T[t])
 print(f"sample {j+1:2d}: {''.join(r)}")