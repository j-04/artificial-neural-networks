package com.dragosh;

import com.dragosh.teacher.BackPropagationTeacher;

/**
 * @author Dragosh Sergey (dragoshs.j@yahoo.com)
 * @since 28-05-2020 (dd-mm-yyyy)
 */
public class App 
{
    public static void main( String[] args )
    {
        BackPropagationTeacher backPropagationTeacher = new BackPropagationTeacher();
        backPropagationTeacher.start();
    }
}
