## Пояснения к формулам EIDM

1. **Условие существования равновесия**  
   $$T < \frac{1}{q}$$  
   Гарантирует, что знаменатель в формулах скорости и расстояния \(1 - qT\) остаётся положительным.

2. **Уравнение равновесия EIDM**  
   $$
   \left(\frac{v_e}{v_0}\right)^{\delta}
   \;+\;
   \left[\frac{q\,(s_0 + T\,v_e)}{v_e}\right]^{2}
   = 1
   $$  
   Баланс свободного разгона (первый член) и поддержания дистанции \(s_e = s_0 + T\,v_e\) (второй член).

3. **Равновесный интервал между машинами**  
   $$s_e = \frac{v_e}{q}$$  
   Поток \(q\) — число машин в секунду, поэтому расстояние между ними \(s_e\) равно скорости, делённой на поток.

4. **Плотность**  
   $$\rho = \frac{1}{s_e}$$  
   Обратная величина интервала \(s_e\): сколько машин на метр дороги.

5. **Коэффициент \(A\)**  
   $$
   A = -\,a\;\Biggl[\;\frac{\delta}{v_e}\Bigl(\frac{v_e}{v_0}\Bigr)^{\!\delta}
   \;+\;\frac{2\,T\,(s_0 + T\,v_e)}{s_e^{2}}\Biggr]
   $$  
   Чувствительность ускорения по собственной скорости \(v\): вклад свободного разгона и дистанционной составляющей.

6. **Коэффициент \(B\)**  
   $$
   B = \frac{a\,v_e\,(s_0 + T\,v_e)}{\sqrt{a\,b}\;s_e^{2}}
   $$  
   Чувствительность ускорения по относительной скорости \(\Delta v\).

7. **Коэффициент \(C\)**  
   $$
   C = \frac{2\,a\,(s_0 + T\,v_e)^{2}}{s_e^{3}}
   $$  
   Чувствительность ускорения по дистанции \(s\).

8. **Суммарный коэффициент \(D\)**  
   $$D = A + 2\,B$$  
   Комбинация \(A\) и \(B\) в характеристическом уравнении для продольных возмущений.

9. **Частота колебаний \(\omega\)**  
   $$
   \omega = \sqrt{\frac{D^{2} + \sqrt{D^{4} + 16\,C^{2}}}{2}}
   $$  
   «Натуральная» частота системы без запаздывания.

10. **Показатель устойчивости (аппрокс.)**  
    $$
    \Re\lambda = D\,\cos(\omega\tau)\;-\;\omega\,\sin(\omega\tau)
    $$  
    При \(\Re\lambda < 0\) — устойчивость; \(\Re\lambda > 0\) — рост stop-and-go волн.
