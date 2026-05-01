function [yout,ystring] = LibraryThree(yin,dy,A,polynomial,usesin,seconddiffX,seconddiffY,seconddiffZ,dim,dynoise,Tt)

T = size(yin,1);
n = size(yin,2)/3;
ind = 1;
h=0.01;

omega = normrnd(0,sqrt(h),Tt,n*3);

if(polynomial>=1)
    for i = 1:n
        yout((i-1)*T+1:i*T,ind) = yin(:,3*i-2);
        yout((i-1)*T+1:i*T,ind+1) = yin(:,3*i-1);
        yout((i-1)*T+1:i*T,ind+2) = yin(:,3*i);
    end
ystring{ind} = 'x';
ystring{ind+1} = 'y';
ystring{ind+2} = 'z';
ind = ind+3;
end

if (polynomial>=2)
    for i = 1:n
        yout((i-1)*T+1:i*T,ind) = yin(:,3*i-2).^2;
        yout((i-1)*T+1:i*T,ind+1) = yin(:,3*i-1).^2;
        yout((i-1)*T+1:i*T,ind+2) = yin(:,3*i).^2;
        yout((i-1)*T+1:i*T,ind+3) = yin(:,3*i-2).*yin(:,3*i-1);
        yout((i-1)*T+1:i*T,ind+4) = yin(:,3*i-2).*yin(:,3*i);
        yout((i-1)*T+1:i*T,ind+5) = yin(:,3*i-1).*yin(:,3*i);
    end
ystring{ind} = 'x2';
ystring{ind+1} = 'y2';
ystring{ind+2} = 'z2';
ystring{ind+3} = 'xy';
ystring{ind+4} = 'xz';
ystring{ind+5} = 'yz';
ind = ind+6;
end

if (polynomial>=3)
    for i = 1:n
        yout((i-1)*T+1:i*T,ind) = yin(:,3*i-2).^3;
        yout((i-1)*T+1:i*T,ind+1) = yin(:,3*i-1).^3;
        yout((i-1)*T+1:i*T,ind+2) = yin(:,3*i).^3;
        yout((i-1)*T+1:i*T,ind+3) = (yin(:,3*i-2).^2).*yin(:,3*i-1); %xxy
        yout((i-1)*T+1:i*T,ind+4) = (yin(:,3*i-2).^2).*yin(:,3*i); %xxz
        yout((i-1)*T+1:i*T,ind+5) = (yin(:,3*i-1).^2).*yin(:,3*i-2); %yyx
        yout((i-1)*T+1:i*T,ind+6) = (yin(:,3*i-1).^2).*yin(:,3*i); %yyz
        yout((i-1)*T+1:i*T,ind+7) = (yin(:,3*i).^2).*yin(:,3*i-2); %zzx
        yout((i-1)*T+1:i*T,ind+8) = (yin(:,3*i).^2).*yin(:,3*i-1); %zzy
        yout((i-1)*T+1:i*T,ind+9) = yin(:,3*i-2).*yin(:,3*i-1).*yin(:,3*i); %xyz
    end
ystring{ind} = 'x3';
ystring{ind+1} = 'y3';
ystring{ind+2} = 'z3';
ystring{ind+3} = 'xxy';
ystring{ind+4} = 'xxz';
ystring{ind+5} = 'yyx';
ystring{ind+6} = 'yyz';
ystring{ind+7} = 'zzx';
ystring{ind+8} = 'zzy';
ystring{ind+9} = 'xyz';
ind = ind+10;
end

if (polynomial>=4)
    for i = 1:n
        yout((i-1)*T+1:i*T,ind) = yin(:,3*i-2).^4;
        yout((i-1)*T+1:i*T,ind+1) = yin(:,3*i-1).^4;
        yout((i-1)*T+1:i*T,ind+2) = yin(:,3*i).^4;
        yout((i-1)*T+1:i*T,ind+3) = (yin(:,3*i-2).^3).*yin(:,3*i-1); %xxxy
        yout((i-1)*T+1:i*T,ind+4) = (yin(:,3*i-2).^3).*yin(:,3*i); %xxxz
        yout((i-1)*T+1:i*T,ind+5) = (yin(:,3*i-1).^3).*yin(:,3*i-2); %yyyx
        yout((i-1)*T+1:i*T,ind+6) = (yin(:,3*i-1).^3).*yin(:,3*i); %yyyz
        yout((i-1)*T+1:i*T,ind+7) = (yin(:,3*i).^3).*yin(:,3*i-2); %zzzx
        yout((i-1)*T+1:i*T,ind+8) = (yin(:,3*i).^3).*yin(:,3*i-1); %zzzy
        yout((i-1)*T+1:i*T,ind+9) = (yin(:,3*i-2).^2).*(yin(:,3*i-1).^2); %xxyy
        yout((i-1)*T+1:i*T,ind+10) = (yin(:,3*i-2).^2).*(yin(:,3*i).^2); %xxzz
        yout((i-1)*T+1:i*T,ind+11) = (yin(:,3*i-1).^2).*(yin(:,3*i).^2); %yyzz
        yout((i-1)*T+1:i*T,ind+12) = (yin(:,3*i-2).^2).*yin(:,3*i-1).*yin(:,3*i); %xxyz
        yout((i-1)*T+1:i*T,ind+13) = (yin(:,3*i-1).^2).*yin(:,3*i-2).*yin(:,3*i); %yyxz
        yout((i-1)*T+1:i*T,ind+14) = (yin(:,3*i).^2).*yin(:,3*i-1).*yin(:,3*i-2); %zzxy
    end
ystring{ind} = 'x4';
ystring{ind+1} = 'y4';
ystring{ind+2} = 'z4';
ystring{ind+3} = 'xxxy';
ystring{ind+4} = 'xxxz';
ystring{ind+5} = 'yyyx';
ystring{ind+6} = 'yyyz';
ystring{ind+7} = 'zzzx';
ystring{ind+8} = 'zzzy';
ystring{ind+9} = 'xxyy';
ystring{ind+10} = 'xxzz';
ystring{ind+11} = 'yyzz';
ystring{ind+12} = 'xxyz';
ystring{ind+13} = 'yyxz';
ystring{ind+14} = 'zzxy';
ind = ind+15;
end


for i = 1:n
    yout((i-1)*T+1:i*T,ind) = exp(yin(:,3*i-2));
    yout((i-1)*T+1:i*T,ind+1) = exp(yin(:,3*i-1)); 
    yout((i-1)*T+1:i*T,ind+2) = exp(yin(:,3*i));
end
ystring{ind} = 'expx';
ystring{ind+1} = 'expy';
ystring{ind+2} = 'expz';
ind = ind+3;


if (usesin == 1)
for i = 1:n
    yout((i-1)*T+1:i*T,ind) = sin(yin(:,3*i-2));
    yout((i-1)*T+1:i*T,ind+1) = sin(yin(:,3*i-1)); 
    yout((i-1)*T+1:i*T,ind+2) = sin(yin(:,3*i));
end
ystring{ind} = 'sinx';
ystring{ind+1} = 'siny';
ystring{ind+2} = 'sinz';
ind = ind+3;
end

if(seconddiffX == 1)
   for i = 1:n
   yout((i-1)*T+1:i*T,ind) = dy(:,3*i-2);
   end
ystring{ind} = 'dX';
ind = ind+1;
end

if(seconddiffY == 1)
   for i = 1:n
   yout((i-1)*T+1:i*T,ind) = dy(:,3*i-1);
   end
ystring{ind} = 'dY';
ind = ind+1;
end

if(seconddiffZ == 1)
   for i = 1:n
   yout((i-1)*T+1:i*T,ind) = dy(:,3*i);
   end
ystring{ind} = 'dZ';
ind = ind+1;
end

if(dynoise == 1)
    for i = 1:n
        yout((i-1)*T+1:i*T,ind) = omega(:,3*i-2);
    end
ystring{ind} = 'dynoise';
ind = ind+1;   
end

%% coupling part 1 
if (dim == 1)
    for i = 1:n
        tmp1 = zeros(T,1);
        tmp2 = zeros(T,1);
        tmp3 = zeros(T,1);
        tmp4 = zeros(T,1);
        tmp5 = zeros(T,1);
        tmp6 = zeros(T,1);
        tmp7 = zeros(T,1);
        tmp8 = zeros(T,1);
        tmp9 = zeros(T,1);
        tmp10 = zeros(T,1);
        tmp11 = zeros(T,1);
        tmp12 = zeros(T,1);
        tmp13 = zeros(T,1);
        tmp14 = zeros(T,1);
        tmp15 = zeros(T,1);
        tmp16 = zeros(T,1);
        tmp17 = zeros(T,1);
        tmp18 = zeros(T,1);
        tmp19 = zeros(T,1);
        tmp20 = zeros(T,1);
        tmp21 = zeros(T,1);
        tmp22 = zeros(T,1);
        tmp23 = zeros(T,1);
        tmp24 = zeros(T,1);
        tmp25 = zeros(T,1);
        tmp26 = zeros(T,1);
        tmp27 = zeros(T,1);
        tmp28 = zeros(T,1);
        tmp29 = zeros(T,1);
        tmp30 = zeros(T,1);
        tmp31 = zeros(T,1);
        tmp32 = zeros(T,1);
        tmp33 = zeros(T,1);
        for j = 1:n
            tmp1 = tmp1 + (yin(:,3*j-2)-yin(:,3*i-2)).*A(i,j); % xj-xi
            tmp2 = tmp2 + sin(yin(:,3*j-2)-yin(:,3*i-2)).*A(i,j); %sin(xj-xi)
            tmp3 = tmp3 + exp(yin(:,3*j-2)-yin(:,3*i-2)).*A(i,j); %exp(xj-xi)
            tmp4 = tmp4 + 1./(1+exp(-(yin(:,3*j-2)-yin(:,3*i-2)))).*A(i,j);%sigmoid
            tmp5 = tmp5 + (exp(yin(:,3*j-2)-yin(:,3*i-2))-exp(-(yin(:,3*j-2)-yin(:,3*i-2))))./(exp(yin(:,3*j-2)-yin(:,3*i-2))+exp(-(yin(:,3*j-2)-yin(:,3*i-2)))).*A(i,j); %tanh
            tmp6 = tmp6 + yin(:,3*j-2).*A(i,j);% xj
            tmp7 = tmp7 + sin(yin(:,3*j-2)).*A(i,j); % sin(xj)
            tmp8 = tmp8 + exp(yin(:,3*j-2)).*A(i,j); % exp(xj)
            tmp9 = tmp9 + 1./(1+exp(-yin(:,3*j-2))).*A(i,j); % sigmoid(xj)
            tmp10 = tmp10 + ((exp(yin(:,3*j-2))-exp(-yin(:,3*j-2)))./(exp(yin(:,3*j-2))+exp(-yin(:,3*j-2)))).*A(i,j); % tanh(xj)
%             tmp11 = tmp11 + (1./(1+exp(-2.*yin(:,3*j-2)))).*A(i,j);
%             tmp12 = tmp12 + (1./(1+exp(-3.*yin(:,3*j-2)))).*A(i,j);
%             tmp13 = tmp13 + (1./(1+exp(-4.*yin(:,3*j-2)))).*A(i,j);
            tmp11 = tmp11 + (1./(1+exp(-5.*yin(:,3*j-2)))).*A(i,j);
%             tmp15 = tmp15 + (1./(1+exp(-6.*yin(:,3*j-2)))).*A(i,j);
%             tmp16 = tmp16 + (1./(1+exp(-7.*yin(:,3*j-2)))).*A(i,j);
%             tmp17 = tmp17 + (1./(1+exp(-8.*yin(:,3*j-2)))).*A(i,j);
%             tmp18 = tmp18 + (1./(1+exp(-9.*yin(:,3*j-2)))).*A(i,j);
            tmp12 = tmp12 + (1./(1+exp(-10.*yin(:,3*j-2)))).*A(i,j);
            tmp13 = tmp13 + (1./(1+exp(-1.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp21 = tmp21 + (1./(1+exp(-2.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp22 = tmp22 + (1./(1+exp(-3.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp23 = tmp23 + (1./(1+exp(-4.*(yin(:,3*j-2)-1)))).*A(i,j);
            tmp14 = tmp14 + (1./(1+exp(-5.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp25 = tmp25 + (1./(1+exp(-6.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp26 = tmp26 + (1./(1+exp(-7.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp27 = tmp27 + (1./(1+exp(-8.*(yin(:,3*j-2)-1)))).*A(i,j);
%             tmp28 = tmp28 + (1./(1+exp(-9.*(yin(:,3*j-2)-1)))).*A(i,j);
            tmp15 = tmp15 + (1./(1+exp(-10.*(yin(:,3*j-2)-1)))).*A(i,j);%%%%%%%%10
            tmp16 = tmp16 + (yin(:,3*i-2)./(1+exp(-1.*(yin(:,3*j-2)-1)))).*A(i,j);
            tmp17 = tmp17 + (yin(:,3*i-2)./(1+exp(-5.*(yin(:,3*j-2)-1)))).*A(i,j);
            tmp18 = tmp18 + (yin(:,3*i-2)./(1+exp(-10.*(yin(:,3*j-2)-1)))).*A(i,j);%%%%%%%10
            tmp19 = tmp19 + (yin(:,3*i-2)./(1+exp(-1.*yin(:,3*j-2)))).*A(i,j);
            tmp20 = tmp20 + (yin(:,3*i-2)./(1+exp(-5.*yin(:,3*j-2)))).*A(i,j);
            tmp21 = tmp21 + (yin(:,3*i-2)./(1+exp(-10.*yin(:,3*j-2)))).*A(i,j);
            tmp22 = tmp22 + (1./(1+exp(-1.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp23 = tmp23 + (1./(1+exp(-5.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp24 = tmp24 + (1./(1+exp(-10.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp25 = tmp25 + (1./(1+exp(-1.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
            tmp26 = tmp26 + (1./(1+exp(-5.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
            tmp27 = tmp27 + (1./(1+exp(-10.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
            tmp28 = tmp28 + (yin(:,3*i-2)./(1+exp(-1.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp29 = tmp29 + (yin(:,3*i-2)./(1+exp(-5.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp30 = tmp30 + (yin(:,3*i-2)./(1+exp(-10.*(yin(:,3*j-2)-yin(:,3*i-2))))).*A(i,j);
            tmp31 = tmp31 + (yin(:,3*i-2)./(1+exp(-1.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
            tmp32 = tmp32 + (yin(:,3*i-2)./(1+exp(-5.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
            tmp33 = tmp33 + (yin(:,3*i-2)./(1+exp(-10.*((yin(:,3*j-2)-yin(:,3*i-2))-1)))).*A(i,j);
        end
        yout((i-1)*T+1:i*T,ind) = tmp1;
        yout((i-1)*T+1:i*T,ind+1) = tmp2;
        yout((i-1)*T+1:i*T,ind+2) = tmp3;
        yout((i-1)*T+1:i*T,ind+3) = tmp4;
        yout((i-1)*T+1:i*T,ind+4) = tmp5;
        yout((i-1)*T+1:i*T,ind+5) = tmp6;
        yout((i-1)*T+1:i*T,ind+6) = tmp7;
        yout((i-1)*T+1:i*T,ind+7) = tmp8;
        yout((i-1)*T+1:i*T,ind+8) = tmp9;
        yout((i-1)*T+1:i*T,ind+9) = tmp10;
        yout((i-1)*T+1:i*T,ind+10) = tmp11;
        yout((i-1)*T+1:i*T,ind+11) = tmp12;
        yout((i-1)*T+1:i*T,ind+12) = tmp13;
        yout((i-1)*T+1:i*T,ind+13) = tmp14;
        yout((i-1)*T+1:i*T,ind+14) = tmp15;
        yout((i-1)*T+1:i*T,ind+15) = tmp16;
        yout((i-1)*T+1:i*T,ind+16) = tmp17;
        yout((i-1)*T+1:i*T,ind+17) = tmp18;
        yout((i-1)*T+1:i*T,ind+18) = tmp19;
        yout((i-1)*T+1:i*T,ind+19) = tmp20;
        yout((i-1)*T+1:i*T,ind+20) = tmp21;
        yout((i-1)*T+1:i*T,ind+21) = tmp22;
        yout((i-1)*T+1:i*T,ind+22) = tmp23;
        yout((i-1)*T+1:i*T,ind+23) = tmp24;
        yout((i-1)*T+1:i*T,ind+24) = tmp25;
        yout((i-1)*T+1:i*T,ind+25) = tmp26;
        yout((i-1)*T+1:i*T,ind+26) = tmp27;
        yout((i-1)*T+1:i*T,ind+27) = tmp28;
        yout((i-1)*T+1:i*T,ind+28) = tmp29;
        yout((i-1)*T+1:i*T,ind+29) = tmp30;
        yout((i-1)*T+1:i*T,ind+30) = tmp31;
        yout((i-1)*T+1:i*T,ind+31) = tmp32;
        yout((i-1)*T+1:i*T,ind+32) = tmp33;
    end
    ystring{ind} = 'xjMINUSxi';
    ystring{ind+1} = 'sinxjMINUSxi';
    ystring{ind+2} = 'expxjMINUSxi';
    ystring{ind+3} = 'sigmoidxjMINUSxi10';
    ystring{ind+4} = 'tanhxjMINUSxi';
    ystring{ind+5} = 'xj';
    ystring{ind+6} = 'sinxj';
    ystring{ind+7} = 'expxj';
    ystring{ind+8} = 'sigmoidxj10';
    ystring{ind+9} = 'tanhxj';
    ystring{ind+10} = 'sigmoidxj50';
    ystring{ind+11} = 'sigmoidxj100';
    ystring{ind+12} = 'sigmoidxj11';
    ystring{ind+13} = 'sigmoidxj51';
    ystring{ind+14} = 'sigmoidxj101';
    ystring{ind+15} = 'xisigmoidxj11';
    ystring{ind+16} = 'xisigmoidxj51';
    ystring{ind+17} = 'xisigmoidxj101';
    ystring{ind+18} = 'xisigmoidxj10';
    ystring{ind+19} = 'xisigmoidxj50';
    ystring{ind+20} = 'xisigmoidxj100';
    ystring{ind+21} = 'sigmoidxjxi10';
    ystring{ind+22} = 'sigmoidxjxi50';
    ystring{ind+23} = 'sigmoidxjxi100';
    ystring{ind+24} = 'sigmoidxjxi11';
    ystring{ind+25} = 'sigmoidxjxi51';
    ystring{ind+26} = 'sigmoidxjxi101';
    ystring{ind+27} = 'xisigmoidxjxi10';
    ystring{ind+28} = 'xisigmoidxjxi50';
    ystring{ind+29} = 'xisigmoidxjxi100';
    ystring{ind+30} = 'xisigmoidxjxi11';
    ystring{ind+31} = 'xisigmoidxjxi51';
    ystring{ind+32} = 'xisigmoidxjxi101';
    ind = ind+10;%33
end

if (dim == 2)
    for i = 1:n
        tmp1 = zeros(T,1);
        tmp2 = zeros(T,1);
        tmp3 = zeros(T,1);
        tmp4 = zeros(T,1);
        tmp5 = zeros(T,1);
        tmp6 = zeros(T,1);
        tmp7 = zeros(T,1);
        for j = 1:n
            tmp1 = tmp1 + (yin(:,3*j-1)-yin(:,3*i-1)).*A(i,j); 
            tmp2 = tmp2 + sin(yin(:,3*j-1)-yin(:,3*i-1)).*A(i,j); 
            tmp3 = tmp3 + exp(yin(:,3*j-1)-yin(:,3*i-1)).*A(i,j); 
            tmp4 = tmp4 + 1./(1+exp(-(yin(:,3*j-1)-yin(:,3*i-1)))).*A(i,j);%sigmoid
            tmp5 = tmp5 + (exp(yin(:,3*j-1)-yin(:,3*i-1))-exp(-(yin(:,3*j-1)-yin(:,3*i-1))))./(exp(yin(:,3*j-1)-yin(:,3*i-1))+exp(-(yin(:,3*j-1)-yin(:,3*i-1)))).*A(i,j); %tanh
            tmp6 = tmp6 + (1./(1+exp(-10.*(yin(:,3*j-1)-1)))).*A(i,j);
            tmp7 = tmp7 + (yin(:,3*i-1)./(1+exp(-10.*(yin(:,3*j-1)-1)))).*A(i,j);
        end
        yout((i-1)*T+1:i*T,ind) = tmp1;
        yout((i-1)*T+1:i*T,ind+1) = tmp2;
        yout((i-1)*T+1:i*T,ind+2) = tmp3;
        yout((i-1)*T+1:i*T,ind+3) = tmp4;
        yout((i-1)*T+1:i*T,ind+4) = tmp5;
        yout((i-1)*T+1:i*T,ind+5) = tmp6;
        yout((i-1)*T+1:i*T,ind+6) = tmp7;
    end
    ystring{ind} = 'yjMINUSyi';
    ystring{ind+1} = 'sinyjMINUSyi';
    ystring{ind+2} = 'expyjMINUSyi';
    ystring{ind+3} = 'sigmoidyjMINUSyi';
    ystring{ind+4} = 'tanhyjMINUSyi';
    ystring{ind+5} = 'true1';
    ystring{ind+6} = 'true2';
    ind = ind+7;
end

if (dim == 3)
    for i = 1:n
        tmp1 = zeros(T,1);
        tmp2 = zeros(T,1);
        tmp3 = zeros(T,1);
        tmp4 = zeros(T,1);
        tmp5 = zeros(T,1);
        tmp6 = zeros(T,1);
        tmp7 = zeros(T,1);
        for j = 1:n
            tmp1 = tmp1 + (yin(:,3*j)-yin(:,3*i)).*A(i,j); % xj-xi
            tmp2 = tmp2 + sin(yin(:,3*j)-yin(:,3*i)).*A(i,j); %sin(xj-xi)
            tmp3 = tmp3 + exp(yin(:,3*j)-yin(:,3*i)).*A(i,j); %exp(xj-xi)
            tmp4 = tmp4 + 1./(1+exp(-(yin(:,3*j)-yin(:,3*i)))).*A(i,j);%sigmoid
            tmp5 = tmp5 + (exp(yin(:,3*j)-yin(:,3*i))-exp(-(yin(:,3*j)-yin(:,3*i))))./(exp(yin(:,3*j)-yin(:,3*i))+exp(-(yin(:,3*j)-yin(:,3*i)))).*A(i,j); %tanh
            tmp6 = tmp6 + (1./(1+exp(-10.*(yin(:,3*j)-1)))).*A(i,j);
            tmp7 = tmp7 + (yin(:,3*i)./(1+exp(-10.*(yin(:,3*j)-1)))).*A(i,j);
        end
        yout((i-1)*T+1:i*T,ind) = tmp1;
        yout((i-1)*T+1:i*T,ind+1) = tmp2;
        yout((i-1)*T+1:i*T,ind+2) = tmp3;
        yout((i-1)*T+1:i*T,ind+3) = tmp4;
        yout((i-1)*T+1:i*T,ind+4) = tmp5;
        yout((i-1)*T+1:i*T,ind+5) = tmp6;
        yout((i-1)*T+1:i*T,ind+6) = tmp7;
    end
    ystring{ind} = 'zjMINUSzi';
    ystring{ind+1} = 'sinzjMINUSzi';
    ystring{ind+2} = 'expzjMINUSzi';
    ystring{ind+3} = 'sigmoidzjMINUSzi';
    ystring{ind+4} = 'tanhzjMINUSzi';
    ystring{ind+5} = 'true1';
    ystring{ind+6} = 'true2';
    %ind = ind+7;
end


end

