%the true equation for HR network dynamics
function dydt = hindmarshrose(x,A,n,gc)
dydt = zeros(3*n,1);
%parameters
a=1;
b=3;
c=1;
d=5;
s=4;
r=0.004;
p0=-1.6; 
Iext=3.24;
Vsyn1=2;
Vsyn2=-1.5;
k = -1;
for i=1:n
    tmp = 0;
    for j=1:n
        if A(i,j)>=0
        tmp = tmp +gc*(Vsyn1-x(3*i-2))*A(i,j)*(1/(1+exp(k*(x(3*j-2)))));
        else
            tmp = tmp +gc*(Vsyn2-x(3*i-2))*abs(A(i,j))*(1/(1+exp(k*(x(3*j-2)-1))));
        end
    end
    dydt(3*i-2)=x(3*i-1)-a*x(3*i-2)^3+b*x(3*i-2)^2-x(3*i)+Iext+tmp;
    dydt(3*i-1)=c-d*x(3*i-2)^2-x(3*i-1);
    dydt(3*i)=r*(s*(x(3*i-2)-p0)-x(3*i));
end
end


