/*
 * ATmega128 - 가변저항 ADC 읽기 -> LCD 표시 + LED PWM 밝기 제어
 *
 * 동작 (영상 기준)
 *   LCD 1줄: "ADC:100 V:1.960V"   (ADC 8비트 0~255, 전압 = ADC*5/255)
 *   LCD 2줄: "DUTY: 39%"          (듀티 = ADC*100/255, 소수점 버림)
 *   가변저항을 돌리면 PWM LED 밝기가 0% ~ 100% 로 변함
 *
 * 핀 배치 (보드 배선에 맞게 아래 #define 만 바꾸면 됨)
 *   가변저항  : PF0 (ADC0)
 *   PWM LED   : PB7 (OC2, Timer2 Fast PWM)
 *   LCD 데이터 : PORTC (8비트 모드, D0~D7 = PC0~PC7)
 *   LCD 제어   : PG0 = RS, PG1 = RW, PG2 = E
 *
 * 클럭 16MHz, AVCC 기준 5V
 */
#ifndef F_CPU
#define F_CPU 16000000UL
#endif

#include <avr/io.h>
#include <util/delay.h>
#include <stdio.h>

/* ---------------- 핀 설정 ---------------- */
#define ADC_CH        0          /* 가변저항이 연결된 ADC 채널 (PF0) */

#define LCD_DATA_PORT PORTC
#define LCD_DATA_DDR  DDRC
#define LCD_CTRL_PORT PORTG
#define LCD_CTRL_DDR  DDRG
#define LCD_RS        0
#define LCD_RW        1
#define LCD_E         2

/* ---------------- LCD (HD44780, 8비트) ---------------- */
static void lcd_pulse(void)
{
    LCD_CTRL_PORT |= (1 << LCD_E);
    _delay_us(1);
    LCD_CTRL_PORT &= ~(1 << LCD_E);
    _delay_us(50);
}

static void lcd_cmd(unsigned char c)
{
    LCD_CTRL_PORT &= ~((1 << LCD_RS) | (1 << LCD_RW));
    LCD_DATA_PORT = c;
    lcd_pulse();
    if (c == 0x01 || c == 0x02) _delay_ms(2);
}

static void lcd_data(unsigned char d)
{
    LCD_CTRL_PORT |= (1 << LCD_RS);
    LCD_CTRL_PORT &= ~(1 << LCD_RW);
    LCD_DATA_PORT = d;
    lcd_pulse();
}

static void lcd_init(void)
{
    LCD_DATA_DDR = 0xFF;
    LCD_CTRL_DDR |= (1 << LCD_RS) | (1 << LCD_RW) | (1 << LCD_E);
    LCD_CTRL_PORT &= ~((1 << LCD_RS) | (1 << LCD_RW) | (1 << LCD_E));
    _delay_ms(50);
    lcd_cmd(0x38); _delay_ms(5);   /* 8비트, 2줄, 5x8 */
    lcd_cmd(0x38); _delay_us(150);
    lcd_cmd(0x38);
    lcd_cmd(0x0C);                 /* 화면 ON, 커서 OFF */
    lcd_cmd(0x06);                 /* 입력 모드: 오른쪽 이동 */
    lcd_cmd(0x01);                 /* 화면 지우기 */
}

static void lcd_goto(unsigned char row, unsigned char col)
{
    lcd_cmd(0x80 | ((row ? 0x40 : 0x00) + col));
}

static void lcd_puts(const char *s)
{
    while (*s) lcd_data(*s++);
}

/* ---------------- ADC ---------------- */
static void adc_init(void)
{
    DDRF &= ~(1 << ADC_CH);
    /* AVCC 기준, 왼쪽 정렬(ADLAR) -> ADCH 만 읽으면 8비트(0~255) */
    ADMUX  = (1 << REFS0) | (1 << ADLAR) | (ADC_CH & 0x07);
    /* ADC 활성화, 분주 128 (16MHz/128 = 125kHz) */
    ADCSRA = (1 << ADEN) | (1 << ADPS2) | (1 << ADPS1) | (1 << ADPS0);
}

static unsigned char adc_read(void)
{
    ADCSRA |= (1 << ADSC);
    while (ADCSRA & (1 << ADSC));
    return ADCH;
}

/* ---------------- PWM (Timer2, OC2 = PB7) ---------------- */
static void pwm_init(void)
{
    DDRB |= (1 << PB7);
    /* Fast PWM, 비반전, 분주 64 -> 약 976Hz */
    TCCR2 = (1 << WGM21) | (1 << WGM20) | (1 << COM21) | (1 << CS21) | (1 << CS20);
    OCR2 = 0;
}

static void pwm_set(unsigned char v)
{
    if (v == 0) {
        /* Fast PWM 은 OCR=0 이어도 짧은 펄스가 나오므로 완전히 끔 */
        TCCR2 &= ~(1 << COM21);
        PORTB &= ~(1 << PB7);
    } else {
        TCCR2 |= (1 << COM21);
        OCR2 = v;
    }
}

/* ---------------- main ---------------- */
int main(void)
{
    char line[17];
    unsigned char adc;
    unsigned int mv, duty;

    lcd_init();
    adc_init();
    pwm_init();

    while (1) {
        adc  = adc_read();
        mv   = (unsigned long)adc * 5000UL / 255UL;   /* 0 ~ 5000 mV */
        duty = (unsigned int)adc * 100U / 255U;       /* 0 ~ 100 %  */

        pwm_set(adc);

        sprintf(line, "ADC:%3u V:%u.%03uV", adc, mv / 1000, mv % 1000);
        lcd_goto(0, 0);
        lcd_puts(line);

        sprintf(line, "DUTY:%3u%%       ", duty);
        lcd_goto(1, 0);
        lcd_puts(line);

        _delay_ms(100);
    }
}
